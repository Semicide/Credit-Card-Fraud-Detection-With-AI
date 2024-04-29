from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
import numpy as np
from flask_sqlalchemy import SQLAlchemy
import datetime
import haversine as hs
#DB connection
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres.olwlkyaowkgdrbonvxpz:KdHg2RbLxjxhRlL7@aws-0-eu-central-1.pooler.supabase.com:5432/postgres'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define database models
class Client(db.Model):
    cl_id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    homelocationX = db.Column(db.Float)
    homelocationY = db.Column(db.Float)

class Transaction(db.Model):
    tr_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    tr_date = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    cl_id = db.Column(db.Integer, db.ForeignKey('client.cl_id'))
    retailer = db.Column(db.Integer, nullable=False)
    used_chip = db.Column(db.Integer)
    used_pin = db.Column(db.Integer)
    tr_price = db.Column(db.Float,nullable=False)
    online_order = db.Column(db.Integer)
    transactionlocationX = db.Column(db.Float)
    transactionlocationY = db.Column(db.Float)

# Create database tables within Flask application context
with app.app_context():
    db.create_all()

# Modeli yükle
model = load_model("balanced_model.keras")

@app.route('/')
def index():
    clients = Client.query.all()
    transactions = Transaction.query.all()
    return render_template('index.html', clients=clients, transactions=transactions)

@app.route('/predict', methods=['GET'])
def predict():
    client_id = request.form.get('client_id')
    transaction_id = request.form.get('transaction_id')
    # Tahmin sonucunu döndür ve index.html'e gönder
    return render_template('index.html')


@app.route('/add_client', methods=['GET', 'POST'])
def add_client():
    if request.method == 'POST':
        name = request.form['name']
        homelocationX = request.form['homelocationX']
        homelocationY = request.form['homelocationY']

        new_client = Client(name=name, homelocationX=homelocationX, homelocationY=homelocationY)
        db.session.add(new_client)
        db.session.commit()

        return redirect(url_for('index'))

    return render_template('client_form.html')


@app.route('/add_transaction', methods=['GET', 'POST'])
def add_transaction():
    if request.method == 'POST':
        client_id = request.form['client_id']
        retailer = request.form['retailer']
        used_chip = request.form['used_chip']
        used_pin = request.form['used_pin']
        online_order = request.form['online_order']
        tr_price=request.form['tr_price']
        transactionlocationX = request.form['transactionlocationX']
        transactionlocationY = request.form['transactionlocationY']

        new_transaction = Transaction(cl_id=client_id, retailer=retailer, used_chip=used_chip, used_pin=used_pin,
                                      online_order=online_order, transactionlocationX=transactionlocationX,
                                      transactionlocationY=transactionlocationY, tr_price=tr_price)
        db.session.add(new_transaction)
        db.session.commit()

        return redirect(url_for('index'))
    clients = Client.query.all()  # Fetch all clients
    return render_template('transaction_form.html', clients=clients)

@app.route('/clients')
def clients():
    clients = Client.query.all()
    return render_template('clients.html', clients=clients)

@app.route('/transactions/<int:client_id>')
def transactions(client_id):
    client = Client.query.get(client_id)
    transactions = Transaction.query.filter_by(cl_id=client_id).all()
    return render_template('transactions.html', client=client, transactions=transactions)

@app.route('/transaction_perdict/<int:client_id>/<int:transaction_id>', methods=['GET', 'POST'])
def predict_transaction(transaction_id,client_id):
    # Fetch the selected client and transaction
    client = Client.query.get(client_id)
    transaction = Transaction.query.get(transaction_id)

    if client and transaction:
        # Calculate distance from home using Haversine formula
        distance_from_home = hs.haversine((client.homelocationX, client.homelocationY),
                                          (transaction.transactionlocationX, transaction.transactionlocationY))

        # Fetch previous transactions for the client
        previous_transactions = Transaction.query.filter_by(cl_id=client_id).order_by(Transaction.tr_date.desc()).all()

        # Calculate distance from last transaction if there are previous transactions
        if len(previous_transactions)>1:
            last_transaction = previous_transactions[1]
            distance_from_last_transaction = hs.haversine((last_transaction.transactionlocationX, last_transaction.transactionlocationY),
                                                           (transaction.transactionlocationX, transaction.transactionlocationY))
        else:
            # If no previous transactions, set distance_from_last_transaction to 0
            distance_from_last_transaction = 0

        # Calculate median purchase price
        all_transaction_prices = [prev_transaction.tr_price for prev_transaction in previous_transactions]
        median_purchase_price = np.median(all_transaction_prices) if all_transaction_prices else 0

        # Calculate ratio to median purchase price
        ratio_to_median_purchase_price = transaction.tr_price / median_purchase_price if median_purchase_price != 0 else 0
        retailers = [prev_transaction.retailer for prev_transaction in previous_transactions]
        if transaction.retailer in retailers:
            repeat_purchase = 1
            if retailers.count(transaction.retailer) == 1:
                repeat_purchase = 0
        else:
            repeat_purchase = 0
        # Extract features from client and transaction data
        feature1 = distance_from_home
        feature2 = distance_from_last_transaction
        feature3 = ratio_to_median_purchase_price
        feature4 = repeat_purchase
        feature5 = transaction.used_chip
        feature6 = transaction.used_pin
        feature7 = transaction.online_order
        features1 = {
            "Distance from home": distance_from_home,
            "Distance from last transaction": distance_from_last_transaction,
            "Ratio to median purchase price": ratio_to_median_purchase_price,
            "repeat_purchase": repeat_purchase,
            "Used chip": transaction.used_chip,
            "Used pin": transaction.used_pin,
            "Online order": transaction.online_order,
            "median_purchase_price":median_purchase_price,

        }

        # Scale the features
        features = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7]])

        # Make prediction
        prediction = model.predict(features)

        if prediction >= 0.5:
            prediction_result = "Fraud"
        else:
            prediction_result = "Not Fraud"

        return render_template('predict_transaction.html', prediction=prediction_result, clients=Client.query.all(),
                               transactions=Transaction.query.all(),features=features1)
    else:
        # Handle error when client or transaction is not found
        prediction_result = "Client or transaction not found"

    return render_template('clients.html', prediction=prediction_result)
if __name__ == '__main__':
    app.run(debug=True)