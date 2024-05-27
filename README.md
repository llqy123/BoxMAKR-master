
### Requirements
- Python == 3.6
- PyTorch >= 0.4

### Running
- Preprocess

```
python src/preprocess.py --dataset movie    # for MovieLens-1M dataset
python src/preprocess.py --dataset book     # for Book-Crossing dataset    
python src/preprocess.py --dataset music    # for Last.FM dataset  
```
- Train and evaluate in `src`(Need to tune the parameters):


```
python src/main_movie.py
python src/main_book.py
python src/main_music.py
```
