<h2 align="center">
Knowledge Graph-Enhanced Recommendation with Box Embeddings
</h2>

<p align="center">
    <img src="https://img.shields.io/badge/version-1.0.1-blue">
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white">
    <a href="http://cips-cl.org/static/CCL2024/index.html"><img src="https://img.shields.io/badge/CCL-2024-%23bd9f65?labelColor=%2377BBDD&color=3388bb"></a>
</p>

This repository is the official implementation of ["Knowledge Graph-Enhanced Recommendation with Box Embeddings"] accepted by CCL 2024.
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
