# Using this code


Launch the vector database:
```
sudo docker compose -f compose.yml up
```

Ingest the files you want to search:
```
python run.py --root <path to files>
```

After that's done, you can search with CLIP:

```
python run.py --search "<search string>" [--search_depth <integer search depth>]
```
