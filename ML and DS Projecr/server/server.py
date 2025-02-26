from flask import Flask, request, jsonify
import util
app = Flask(__name__)

@app.route('/get_location_names')
def get_location_names():
    response = jsonify({'locations' : util.get_location_names()})
    response.headers.add('Acces-Control-Allow-Origin', '*')
    
    return response

@app.route('/predict_home_price', methods=['GET', 'POST'])
def predict_home_price():
    sqft = float(request.form['sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response = jsonify({
        'estimated_price' : util.get_estimated_price(location, sqft, bhk, bath)
    })
    
    return response

if __name__ == '__main__':
    print('Запуск Python Flask Server для предсказния цены недвижимости')
    util.load_saved_artifacts()
    app.run()