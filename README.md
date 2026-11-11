# ArtWorks

Turning everyday photos into artwork using classical computer vision. No AI or machine learning, just Python, OpenCV and NumPy.

Group project for the Image Processing module at TU Dublin.

## Team

| Name | Student Number |
|------|----------------|
| Emma Radojcic | C22369396 |
| Amar Jagana | C22423526  |
| Guilherme Crolla | C23336533 |

## About the Project

Each team member develops their own filters in a separate folder, using classical image processing techniques like kernel filtering, thresholding, colour manipulation and blending. All filters are run on test photos, and the results and development process are documented weekly on our blogs.

## Filters

| Filter | Developer | Status |
|--------|-----------|--------|
| Dreamy Glow | Emma | ✅ Done |
| Vignette | Amar | ✅ Done |
| Thermal | Emma | ✅ Done |
| Duotone Popart | Guilherme | ✅ Done |
| Red Pop | Guilherme | ✅ Done |
| Vinatage | Guilherme | ✅ Done |
| Comic Book | Emma | ✅ Done |
| Oil Painting | Emma | ✅ Done |

*More filters will be added as they're allocated each week.*

## Setup

Requires Python 3 with the following packages:

## How to Run

1. Open a team member's notebook in VS Code or Jupyter
2. Make sure your test photos are in that member's `images/input` folder
3. Run all cells from top to bottom
4. Filtered images are saved to `images/output`

Filters can also be run as scripts, e.g. `python emma/comic_book.py Portrait.jpg`.

### Website

Run every filter from the browser:

```
cd website
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000, upload a photo and choose a filter.

## Blogs

- Emma: https://c22369396.wordpress.com/

- Guilherme: https://c23336533.wordpress.com/
