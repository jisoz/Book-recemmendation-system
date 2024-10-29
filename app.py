from flask import  Flask,render_template,request
from fastai.vision.all import *
import pickle
app = Flask(__name__)

popularbooks=pickle.load(open('popularbooks.pkl','rb'))
bookdata=pickle.load(open('bookdata.pkl','rb'))

books=pickle.load(open('books.pkl','rb'))
learn_inf = load_learner('export.pkl')


@app.route('/')
def home():
    return render_template('home.html',
                           bookname=list(popularbooks['Book-Title'].values),
                           bookauthor=list(popularbooks['Book-Author'].values),
                           image=list(popularbooks['Image-URL-M'].values),
                           votes=list(popularbooks['num_rating'].values),
                           rating=list(popularbooks['avg'].values)

                           
                           )

@app.route('/recommend')
def recommend():
      return render_template('recommend.html')

@app.route('/explain')
def explain():
      return render_template('transparency.html')


# @app.route('/recommend_books',methods=['POST'])
# def recommendbks():
#       booknames=[]
#       bookids=[]
#       images=[]
#       authors=[]
#       years=[]
#       input=request.form.get("input")
#       book_bias = learn_inf.model.i_weight.weight
#       idx = books.o2i[str(input)]
#       distances = torch.nn.CosineSimilarity(dim=1)(book_bias, book_bias[idx][None])
#       idx = distances.argsort(descending=True)[:5]
    
#       for i in idx:
#            book=books[i]
#            booknames.append(book)
          
#       for name in booknames:
#            ids=np.where(bookdata['Book-Title']==name)[0][0]
#            bookids.append(ids)
                  
#       for  id in bookids:
#            image=bookdata.iloc[id]['Image-URL-M']
#            author=bookdata.iloc[id]['Book-Author']
#            year=bookdata.iloc[id]["Year-Of-Publication"]
#            images.append(image)
#            authors.append(author)
#            years.append(year)
           
                   
      
#       return render_template("recommend.html",data=booknames,images=images,authors=authors,years=years)
      
@app.route('/recommend_books', methods=['POST'])
def recommendbks():
    booknames = []
    bookids = []
    images = []
    authors = []
    years = []
    input = request.form.get("input").strip().lower()  # Convert input to lowercase and strip whitespace

    book_bias = learn_inf.model.i_weight.weight
    idx = books.o2i.get(str(input))  # Attempt to find exact index for input

    # Check if idx is None, meaning the exact title wasn't found
    if idx is None:
        # Try to find the closest match
        matched_books = [title for title in bookdata['Book-Title'] if input in title.lower()]
        if matched_books:
            # Use the first match if multiple are found
            bookname = matched_books[0]
            idx = books.o2i.get(str(bookname))

            # Calculate distances using the matched index
            distances = torch.nn.CosineSimilarity(dim=1)(book_bias, book_bias[idx][None])
            idx = distances.argsort(descending=True)[:5]

            for i in idx:
                book = books[i]
                booknames.append(book)

            for name in booknames:
                ids = np.where(bookdata['Book-Title'].str.lower() == name.lower())[0]
                if len(ids) > 0:
                    bookids.append(ids[0])  # Get the first match

            for id in bookids:
                image = bookdata.iloc[id]['Image-URL-M']
                author = bookdata.iloc[id]['Book-Author']
                year = bookdata.iloc[id]["Year-Of-Publication"]
                images.append(image)
                authors.append(author)
                years.append(year)

    return render_template("recommend.html", data=booknames, images=images, authors=authors, years=years)

if __name__=="__main__":
    app.run(host="0.0.0.0", port=4000)