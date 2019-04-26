# Quick API guide

## Running and testing

To start a local server, run the following command in terminal after you cd into the directory with the file:`__init__.py` in it. `python __init__.py`

### Image team

To send an image to your local server you run the following command in terminal
` curl --form "fileupload=@download.png" http://127.0.0.1:5000/api/classify `

To send an image to the heroku server replace `127.0.0.1:5000` with `https://clothing-ml-api.herokuapp.com`

Example:
```
curl --form "fileupload=@filename.png" https://clothing-ml-api.herokuapp.com/api/classify
```

See the code in `__init.py__`.
Below is my implementation of the `/api/classify` route. The route classifies an image based on an old model I built.

    @app.route('/api/classify', methods=['POST'])
    def predict():
        data = {'state': False}
        print(request)
        app.logger.info('FILE RECEIVED: %s', request.files)

        img = request.files['fileupload'].read()
        imarr = np.uint8(np.asarray(Image.open(BytesIO(img)).convert('RGB').resize((224,224))))

        trans = transforms.ToTensor()
        imarr = trans(imarr)
        imarr = imarr.unsqueeze(0)
        
        data = predict_img(imarr)
        return jsonify(data)

### NLP team

To send a json to your local server you run the following command in terminal
`curl -X POST -H "Content-Type: application/json" -d '{"test_string":"value1"}'  http://127.0.0.1:5000/api/test_text`


To send a json to the heroku server replace `127.0.0.1:5000` with `https://clothing-ml-api.herokuapp.com`

Example:
`curl -X POST -H "Content-Type: application/json" -d '{"test_string":"value1"}'  https://clothing-ml-api.herokuapp.com/api/testtext`

See below for an example implementation that simply returns the received string

    @app.route('/api/test_text', methods=['POST'])
    def return_text():

        data = {'state': False}
        print(request)
        s = request.get_json()['test_string']
        print(s)

        app.logger.info('STRING RECEIVED: %s', s)
        data['state'] = True
        data['response'] = s
        return jsonify(data)
