import argparse
import hashlib
import mimetypes
import os

from pathlib import Path

import psycopg
import torch

from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, CLIPModel


def init_database():
	connection = get_db_connection()
	cursor = connection.cursor()
	cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
	create_table_statement = """
	CREATE TABLE IF NOT EXISTS files(
	file_path TEXT,
	sha3_hash CHAR(64),
	clip_embedding vector(512)
	);"""
	cursor.execute(create_table_statement)
	connection.commit()
	cursor.close()

def read_in_chunks(file_object, chunk_size=1024):
	while True:
		data = file_object.read(chunk_size)
		if not data:
			break
		yield data

def hash_file(path: Path, chunk_size: int = 65535) -> str:
	hash_fn = hashlib.sha3_256()
	with open(path, "rb") as f:
		for file_chunk in read_in_chunks(f, chunk_size=chunk_size):
			hash_fn.update(file_chunk)
	return str(hash_fn.hexdigest())

def guess_mime_prefix(path):
	try:
		prefix = mimetypes.guess_type(path)[0].split("/")[0]
	except Exception as e:
		prefix = ""
	return prefix

def get_db_connection():
	db_host = os.environ.get("POSTGRES_HOST", "localhost")
	db_user = os.environ.get("POSTGRES_USER", "user")
	db_name = os.environ.get("POSTGRES_NAME", "vectordb")
	db_port = os.environ.get("POSTGRES_PORT", "5432")
	if db_port[0] != ":":
		db_port = ":" + db_port
	db_password = os.environ.get("POSTGRES_PASSWORD", "password")
	db_url = f"postgresql://{db_user}:{db_password}@{db_host}{db_port}/{db_name}"
	connection = psycopg.connect(db_url)
	register_vector(connection)
	return connection


def load_clip_model() -> (CLIPModel, AutoProcessor, AutoTokenizer):
	clip_model = "openai/clip-vit-base-patch32"
	#device = "cuda" if torch.cuda.is_available() else "cpu"
	device = "cpu"

	model = CLIPModel.from_pretrained(clip_model).to(device)
	processor = AutoProcessor.from_pretrained(clip_model)
	tokenizer = AutoTokenizer.from_pretrained(clip_model)

	return model, processor, tokenizer

def ingest(root: Path):
	init_database()
	file_list = [f for f in root.rglob("*") if f.is_file() and guess_mime_prefix(f) == "image"]

	connection = get_db_connection()
	clip_model, clip_processor, _ = load_clip_model()
	for file in tqdm(file_list):
		try:
			cursor = connection.cursor()
			file_hash = hash_file(file)
			select_statement = f"SELECT * FROM files WHERE sha3_hash LIKE '{file_hash}'"
			cursor.execute(select_statement)
			results = cursor.fetchall()
			if len(results) > 0:
				tqdm.write("{file} already in database, continuing")
				continue
			with Image.open(file) as img:
				inputs = clip_processor(images=[img], return_tensors="pt")
				image_features = clip_model.get_image_features(**inputs).detach().numpy().flatten()
			insert_statement = """INSERT INTO files (file_path, sha3_hash, clip_embedding) VALUES (%s, %s, %s);"""
			values = (str(file), file_hash, image_features)
			cursor.execute(insert_statement, values)
			connection.commit()
		except Exception as e:
			tqdm.write(f"Error on {file}: {e}")
			break

def run_clip_query(query_string, search_depth:int = 25):
	connection = get_db_connection()
	clip_model, _, clip_tokenizer = load_clip_model()
	inputs = clip_tokenizer([query_string], padding=True,return_tensors="pt")
	text_features = clip_model.get_text_features(**inputs).detach().numpy().flatten()
	select_statement = f"SELECT (clip_embedding <#> %s) AS similarity, file_path FROM files ORDER BY similarity LIMIT {search_depth}"
	cursor = connection.cursor()
	cursor.execute(select_statement, (text_features, ))
	results = cursor.fetchall()
	for i, r in enumerate(results):
		print(f"{i}: {r[1]}")
	return results

def cli():
	parser = argparse.ArgumentParser()
	parser.add_argument("--root", type=Path)
	parser.add_argument("--search", type=str)
	parser.add_argument("--search_depth", type=int, default=25)

	args = parser.parse_args()

	if args.root is not None:
		print(f"Ingesting photos at {args.root}")
		ingest(args.root)

	if args.search is not None:
		run_clip_query(query_string=args.search, search_depth=args.search_depth)

if __name__ == '__main__':
	load_dotenv()
	cli()
