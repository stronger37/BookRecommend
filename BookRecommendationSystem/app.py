from flask import Flask, render_template, request, redirect, url_for
from recommender import Recommender

app = Flask(__name__)
recommender = Recommender()

@app.route('/')
def index():
    top_books = recommender.get_top_books()
    return render_template('index.html', books=top_books, title="Top Rated Books")

@app.route('/search')
def search():
    query = request.args.get('q')
    if query:
        results = recommender.search_books(query)
        return render_template('index.html', books=results, title=f"Search Results for '{query}'")
    return redirect(url_for('index'))

@app.route('/book/<int:book_id>')
def book_detail(book_id):
    book = recommender.get_book_details(book_id)
    if book:
        # Use ID-based recommendation
        recommendations = recommender.get_recommendations_by_id(book_id)
        # Get reviews
        reviews = recommender.get_reviews(book['Name'])
        return render_template('detail.html', book=book, recommendations=recommendations, reviews=reviews)
    return "Book not found", 404

@app.route('/library')
def library():
    all_books = recommender.get_all_books()
    return render_template('library.html', books=all_books, title="Full Library")

@app.route('/recommend_by_id')
def recommend_by_id_route():
    book_id = request.args.get('book_id')
    if book_id:
        try:
            # Redirect to the book detail page which handles recommendations
            return redirect(url_for('book_detail', book_id=int(book_id)))
        except ValueError:
            return "Invalid Book ID format", 400
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
