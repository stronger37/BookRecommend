This is a web-based book recommendation system built with Flask and Python.

## Setup

1.  **Prerequisites**: Python 3.x installed.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Data Setup**:
    - Place your `books.csv` and `ratings.csv` files in the `data/` directory.
    - Ensure `books.csv` has columns like `Id`, `Name`, `Authors`, `Publisher`, `Rating`, etc.
    - Ensure `ratings.csv` has columns `ID`, `Name`, `Rating`.

## Running the Application

1.  Run the application:
    ```bash
    python app.py
    ```
2.  Open your web browser and navigate to:
    `http://127.0.0.1:5000`

## Features

- **Top Rated Books**: Displays a list of highest-rated books on the home page.
- **Search**: Search for books by title.
- **Recommendations**: When viewing a book, see similar books based on content (Title, Author, Publisher).
