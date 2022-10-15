from flask import Flask, request, render_template, url_for, redirect
from catdogvoice import instant_predict
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        aud = request.files['audio_data']
        val = instant_predict(aud)[0]
        val = 'Cat' if val == 0 else 'Dog'
        return val
    return render_template('home.html')



if __name__ == '__main__':
    app.run(debug=True)
# Cat-0
# Dog-1