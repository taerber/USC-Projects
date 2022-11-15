import os
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory
from werkzeug.utils import secure_filename
import tempfile
import pyrebase
import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
import json
import numpy as np
import fpdf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from pandas_datareader import data
import datetime as dt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
### pip install Pillow #####
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime



config = {
  "apiKey": "AIzaSyBhpkLtMoCmAq-wiztiRkqZ3IMzzv1YEfE",
  "authDomain": "stock-price-ee1ca.firebaseapp.com",
  "databaseURL": "https://stock-price-ee1ca-default-rtdb.firebaseio.com",
  "projectId": "stock-price-ee1ca",
  "storageBucket": "stock-price-ee1ca.appspot.com",
  "messagingSenderId": "874834751520",
  "appId": "1:874834751520:web:e7c2adaddd0e2d03938c45",
  "measurementId": "G-3CP8R3JM9H"
}

firebase = pyrebase.initialize_app(config)

storage = firebase.storage()
#storage1 = firebase.storage().ref("/meta/meta_data_AAPL.txt")
#starsRef = storage1.child("/meta/meta_data_AAPL.txt").put('meta/apple.txt')
starsRef = storage.child("/meta/meta_data_AAPL.txt").download('templates/apple.txt');
starsRef = storage.child("/meta/meta_data_AMZN.txt").download('templates/amazon.txt');
starsRef = storage.child("/meta/meta_data_GOOD.txt").download('templates/google.txt');
starsRef = storage.child("/meta/meta_data_NVDA.txt").download('templates/nvidia.txt');
starsRef = storage.child("/meta/meta_data_TSLA.txt").download('templates/tesla.txt');
UPLOAD_FOLDER = 'templates/uploads'
ALLOWED_EXTENSIONS = {'txt'}

###########################################################################################################################
### spark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("AppleStockPrice").getOrCreate()

from pyspark.sql.types import StructType,StructField, TimestampType, FloatType


schema = StructType( \
    [StructField("date", TimestampType(), True), \
    StructField("open",  FloatType(),True), \
    StructField("high",  FloatType(), True), \
    StructField("low",  FloatType(), True), \
    StructField('close',  FloatType(), True), \
    StructField("volume",  FloatType(), True), \
                    ])

### streaming function
def streaming_data(input_csv):
    streamingDF = (
        spark
            .readStream
            .schema(schema)
            .option("maxFilesPerTrigger", 1)
            .load(input_csv)
    )

    # Stream `streamingDF` while aggregating by date
    streamingActionCountsDF = (
        streamingDF
            .groupBy(
            streamingDF.date
        )
            .count()
    )

    print("The input csv is: " + str(input_csv))
    print("The data is streaming: ")

    return streamingActionCountsDF.isStreaming


from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

## linear regression
def linear_regression(df):
    ###### making result to PDF file

    pdf = fpdf.FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Model Result for Linear Regression: ", ln=1, align="L")

    assembler = VectorAssembler(
        inputCols=['open', 'high', 'low', 'volume'],
        outputCol='features')

    output_data = assembler.transform(df)
    output_data = output_data.select(['features', 'close'])
    test_set, train_set = output_data.randomSplit([0.3, 0.7])

    ## linear regression model
    lr = LinearRegression(featuresCol='features',
                          labelCol='close',
                          maxIter=10,
                          regParam=0.3,
                          elasticNetParam=0.8)

    lr_model = lr.fit(train_set)

    test02 = test_set.select(['features', 'close'])
    test03 = lr_model.transform(test02)
    test03.show(5)

    print("Coefficients of Linear Regression for Close Price: " + str(lr_model.coefficients))

    pdf.cell(200, 10, txt="Coefficients of Linear Regression for Close Price: ", ln=1, align="L")

    pdf.cell(200, 10, txt=str(lr_model.coefficients), ln=1, align="L")

    print("Intercept of Linear Regression for Close Price: " + str(lr_model.intercept))

    pdf.cell(200, 10, txt="Intercept of Linear Regression for Close Price: " + str(lr_model.intercept), ln=1, align="L")

    lr_m = lr_model.summary
    print("MSE: %f" % lr_m.meanSquaredError)
    pdf.cell(200, 10, txt="Linear Regression Test MSE: " + str(lr_m.meanSquaredError), ln=1, align="L")
    pdf.output("lr_output.pdf")
    return lr_m.meanSquaredError

    #### KNN
def KNN_reg(path_on_cloud, K_range):
    # get data from the cloud storage
    path_on_cloud = path_on_cloud
    url = storage.child(path_on_cloud).get_url(None)
    data = pd.read_csv(url)  # read data

    data = data.drop(columns=['time', 'TradeDate', 'date'])
    # getting train and test set
    train_set, test_set = train_test_split(data, test_size=0.3)
    x_train, y_train = train_set.drop(columns=['close']), train_set.iloc[:, 3]
    x_test, y_test = test_set.drop(columns=['close']), test_set.iloc[:, 3]

    # normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaled = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(x_train_scaled)
    x_test_scaled = scaler.fit_transform(x_test)
    x_test = pd.DataFrame(x_test_scaled)

    ###### making result to PDF file

    pdf = fpdf.FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Model Result for KNN: ", ln=1, align="L")

    print("Model Result for KNN:")
    # KNN model selection
    mse_val = []
    for K in range(K_range):
        K = K + 1
        model = neighbors.KNeighborsRegressor(n_neighbors=K)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        error = mean_squared_error(y_test, pred)
        mse_val.append(error)
        print('MSE value for k = ', K, 'is:', error)
        pdf.cell(200, 10, txt="MSE value for k = " + str(K) + " is: " + str(error), ln=1, align="L")

    # # plot MSE
    # x = np.linspace(1, len(mse_val), len(mse_val)).astype('int')
    # y = np.array(mse_val)
    # plt.xlabel('K')
    # plt.ylabel('MSE')
    # plt.plot(x, y)
    # plt.title("Plots of MSEs for KNN Regression")
    # plt.savefig("MSEs.jpg")
    # plt.show()

    # pdf.image("MSEs.jpg")

    # getting the best K
    best_K = mse_val.index(min(mse_val)) + 1
    print("The best number of K nearest neighbors is ", best_K)
    pdf.cell(200, 10, txt="The best number of K nearest neighbors is " + str(best_K), ln=1, align="L")

    best_knn = neighbors.KNeighborsRegressor(n_neighbors=best_K)
    best_knn.fit(x_train, y_train)
    y_pred = best_knn.predict(x_test)

    test_mse = mean_squared_error(y_test, y_pred)
    print("KNN Model Test MSE: ", test_mse)
    pdf.cell(200, 10, txt="KNN Model Test MSE: " + str(test_mse), ln=1, align="L")

    # plt.scatter(y_test.index, y_test, color='red', label='True Value', marker="d")
    # plt.scatter(y_test.index, y_pred, color='blue', label='Prediction', marker="1")
    # plt.xlabel('The index of test data set')
    # plt.ylabel('Stock Close Price')
    # plt.title("Plots of Best KNN model predition")
    # plt.legend()
    # plt.savefig("pred_vs_true.jpg")
    # plt.show()

    # pdf.image("pred_vs_true.jpg")

    pdf.output("knn_output.pdf")

    return test_mse

### xgboost
def xgboost_model(path_on_cloud):

  ### create a pdf for save model result
  pdf = fpdf.FPDF(orientation='P', unit='mm', format='A4')
  pdf.add_page()
  pdf.set_font("Arial", size=12)
  pdf.cell(200, 10, txt="Model Result for XGBoost: ", ln=1, align="L")


  url = storage.child(path_on_cloud).get_url(None)
  df = pd.read_csv(url)
  df = df.drop(columns = ['date'])
  df = df.drop(columns = ['TradeDate'])
  df = df.drop(columns = ['time'])

  train_df , test_df = train_test_split(df, test_size = 0.3)
  y_train = train_df.iloc[:,3]
  x_train = train_df.drop(columns = ['close'])
  y_test = test_df.iloc[:,3]
  x_test = test_df.drop(columns = ['close'])

  train_m = xgb.DMatrix(data = x_train, label=y_train)
  test_m = xgb.DMatrix(data = x_test)
  error = []
  for i in [10 ** i for i in range(-10, 11)]:
      xgb_model = xgb.XGBRegressor(objective = "reg:squarederror", reg_alpha= i)
      cvs = cross_val_score(xgb_model, x_train, y_train, cv = KFold(5, shuffle = True, random_state = 42), scoring = "neg_mean_squared_error")
      error.append((i, np.abs(cvs.mean())))
  error.sort(key=lambda x: x[1])
  print("The Best Alpha =", error[0][0])
  pdf.cell(200, 10, txt="The Best Alpha = " +str(error[0][0]), ln=1, align="L")

  xgb_model = xgb.XGBRegressor(objective = 'reg:squarederror', reg_alpha= error[0][0]).fit(x_train, y_train)
  pred = xgb_model.predict(x_test)
  test_mse = mean_squared_error(y_test, pred)
  print("XGBoost Model Test MSE:", test_mse)
  pdf.cell(200, 10, txt="XGBoost Model Test MSE: " +str(test_mse), ln=1, align="L")


  # fig, ax = plt.subplots(figsize=(10, 10))
  # xgb.plot_tree(xgb_model, ax=ax)
  # plt.title("The Tree Structure of XGBoost Model")
  # plt.savefig("tree_structure.jpg")
  # plt.show()

  # pdf.image("tree_structure.jpg")
  pdf.output("xgboost_output.pdf")

  return test_mse


### LSTM
def LSTM_model(path_on_cloud, num_epoch):
    ### create a pdf for save model result
    #pdf = fpdf.FPDF(orientation='P', unit='mm', format='A4')
    #pdf.add_page()
    #pdf.set_font("Arial", size=12)
    #pdf.cell(200, 10, txt="Model Result for LSTM: ", ln=1, align="L")

    url = storage.child(path_on_cloud).get_url(None)
    df = pd.read_csv(url)

    # plt.plot(range(df.shape[0]), df['close'])
    # plt.xticks(range(0, df.shape[0], 500), df['date'].loc[::500], rotation=45)
    # plt.xlabel('Date', fontsize=12)
    # plt.ylabel('Close Price', fontsize=12)
    # plt.title("Close Price Based on Date")
    # plt.savefig("close.jpg")
    # plt.show()

    # pdf.image("close.jpg")

    df = df.drop(columns=['date'])
    df = df.drop(columns=['TradeDate'])
    df = df.drop(columns=['time'])

    train_df, test_df = train_test_split(df, test_size=0.3)
    train_df_processed = train_df.iloc[:, 3:4].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df_scaled = scaler.fit_transform(train_df_processed)

    features_set = []
    labels = []
    for i in range(60, len(train_df_scaled)):
        features_set.append(train_df_scaled[i - 60:i, 0])
        labels.append(train_df_scaled[i, 0])

    features_set, labels = np.array(features_set), np.array(labels)
    features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(features_set, labels, epochs=num_epoch, batch_size=32)

    test_df_processed = test_df.iloc[:, 3:4].values
    test_df_scaled = scaler.fit_transform(test_df_processed)
    df_total = pd.concat((train_df['close'], test_df['close']), axis=0)
    test_inputs = df_total[len(df_total) - len(test_df) - 60:].values
    test_inputs = test_inputs.reshape(-1, 1)
    test_inputs = scaler.transform(test_inputs)
    test_features = []
    for i in range(60, 80):
        test_features.append(test_inputs[i - 60:i, 0])
    test_features = np.array(test_features)
    test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))

    predictions = model.predict(test_features)
    predictions = scaler.inverse_transform(predictions)

    testScore = mean_squared_error(test_df_processed[60:80], predictions)
    print('Test_mse: ', testScore)
    #pdf.cell(400, 10, txt="LSTM Model Test MSE: " + str(testScore), ln=1, align="L")

    # plt.plot(predictions, color='red', label='Predicted Stock Price')
    # plt.title('Future 60 Days Stock Close Price Prediction')
    # plt.xlabel('Date')
    # plt.ylabel('Stock Price')
    # plt.legend()
    # plt.savefig("prediction.jpg")
    # plt.show()

    # pdf.image("prediction.jpg")
    #pdf.output("LSTM_output.pdf")

    return testScore



##############################function for get image metadata#########################
def meta_image(input_image_path):
    image = Image.open(input_image_path)

    w, h = image.size
    # print("A brief info about this image: ")
    print('Image Width: ', w)
    print('Image Height:', h)
    print()
    exifdata = image.getexif()

    info = []
    for tag_id in exifdata:
        # get the tag name, instead of human unreadable tag id
        tag = TAGS.get(tag_id, tag_id)
        data = exifdata.get(tag_id)
        # decode bytes
        if isinstance(data, bytes):
            data = data.decode()
        info.append(f"{tag:25}: {data}")
        print(f"{tag:25}: {data}")

    #### write metadata info into a txt file

    with open('image_meta.txt', 'w') as f:
        # f.write("A brief info about this image: ")
        f.write('\n')
        f.write('Image Width: ' + str(w))
        f.write('\n')
        f.write('Image Height:' + str(h))
        f.write('\n')
        f.write('\n')
        #         f.write("Metadata about this image: ")
        for line in info:
            f.write(line)
            f.write('\n')

    with open("image_meta.txt", "r") as fd:
        new_line = []
        for line in fd:
            line = line.strip()
            new_line.append(line)

    ## upload and store metadata as txt file in firebase
    path_cloud_image_meta = 'image_meta/image_meta.txt'
    path_local_image_meta = 'image_meta.txt'
    storage.child(path_cloud_image_meta).put(path_local_image_meta)

    url_image_meta = storage.child(path_cloud_image_meta).get_url(None)
    ## function return list, url_link
    ## list of info for image metadata
    ## url link that retrieve from firebase
   




    # datetime object containing current date and time
    now = datetime.now()
 
    #print("now =", now)

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return new_line, url_image_meta, dt_string




###########################################################################################################################


app = Flask(__name__, template_folder='templates', static_folder = 'static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route('/rows')
# def rows():
#     rowNum = request.form.get("rows")
#     return rowNum


@app.route('/')
def index():
    
   # storage.child("user_data/user_home.png").put(filename)
    textF = storage.child("user_data/graph.png").get_url("graph.png")
    return render_template('home.html', textF = textF)

@app.route('/personal')  
def personal():  
    return render_template("personal.html")

# @app.route('/amazon') 
# def amazon():  
#     return render_template("amazon.html")  
# @app.route('/nvidia') 
# def nvidia():  
#     return render_template("nvidia.html")  
# @app.route('/tesla') 
# def tesla():  
#     return render_template("tesla.html")  
 
# @app.route('/google') 
# def google():  
#     return render_template("google.html")  
@app.route('/home') 
def home():
    textF = storage.child("user_data/graph.png").get_url("graph.png")
    return render_template('home.html', textF = textF)  
    #return render_template("home.html")

@app.route('/personal_data', methods = ['POST'])  
def success():  
      
    f = request.files['file']
    filename = secure_filename(f.filename)
    fileN, fileType = os.path.splitext(filename)

    if fileType == ".txt":


        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        file = open(filename,"r")
    
        m = file.read()
        m = [x.strip() for x in m.split(',')]
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))

        storage.child("user_data/user_data.txt").put(filename)
        textF = storage.child("user_data/user_data.txt").get_url("user_data.txt")
        return render_template("personal_data.html", timeN = dt_string, text = m, fileType = fileType, textF = textF)
    else:
    
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))

        storage.child("user_data/user_data.jpg").put(filename)
        imgN = storage.child("user_data/user_data.jpg").get_url("user_data.jpg")
        urllib.request.urlretrieve(imgN, "user_data.jpg")
        new_line, url_image_meta, timeN = meta_image("user_data.jpg")
        return render_template("personal_data.html", imgN = imgN, text = "",
                               list_meta = new_line, url_meta = url_image_meta, timeN = timeN, fileType = fileType)
        #return render_template("personal_data.html", imgN = imgN, text = "")

    # with open(f.filename,"r") as file:
    #     listl=[]
    #     for line in file:
    #     	strip_lines=line.strip()
    #     	listli=strip_lines.split()
    #     	m=listl.append(listli)
        	
    #return render_template("personal_data.html", text = m) #len1 = len1)

###########################################################################################################################
# APPLE CODE #

@app.route('/apple') 
def apple():

    #f = request.files['file']
    #filename = secure_filename(f.filename)
    filename = 'templates/apple.txt'
    file = open(filename,"r")
    #text = []
    m = file.read()
    m = [x.strip() for x in m.split('\n')]
    ## getting raw data
    ## output format: pd DataFrame
    path_on_cloud = "stock_csv/AAPL/raw_AAPL.csv"
    url = storage.child(path_on_cloud).get_url(None)
    apple_data = pd.read_csv(url, header = 0)
    #apple_data1 = apple_data.values.tolist()

  
    ## getting mata data
    ## output format: list
    meta_on_cloud = "meta/meta_data_AAPL.txt"
    url_meta = storage.child(meta_on_cloud).get_url(None)
    file = urllib.request.urlopen(url_meta).read().splitlines()
    meta_list = []
    for line in file:
        decoded_line = line.decode("utf-8")
        meta_list.append(decoded_line)

    ## getting summary statistics from pyspark
    ## output format: pd DataFrame
    stat_on_cloud = "features/summaryApple.csv"
    url_stat = storage.child(stat_on_cloud).get_url(None)
    apple_stat = pd.read_csv((url_stat), usecols=['summary', 'open', 'high', 'low', 'close', 'volume'])
    apple_stat = apple_stat.values.tolist()
    ## getting feature extraction from pyspark
    ## output format: url link
    ## user can clink the link to redirect the txt file
    fea_on_cloud = "features/pca_Apple.txt"
    url_fea = storage.child(fea_on_cloud).get_url(None)

    ## getting pyspark steaming
    ## Output fromat: string
    ##### Sample output should be look like this:
    ## The input csv is: apple_data.csv
    ## The data is streaming:
    ## True

    apple_data.to_csv("apple_data.csv", index=False)
    streaming = streaming_data("apple_data.csv")


    ## linear regression
    apple_df = spark.read.format('csv').option('header', True).schema(schema).load("apple_data.csv")
    lr_mse = linear_regression(apple_df)
    ## make a html for user to click and view the detailed model result in PDF format
    lr_on_cloud = "model_result/Linear_Regression/AAPL_lr_output.pdf"
    lr_url = storage.child(lr_on_cloud).get_url(None)

    ### KNN
    knn_mse = KNN_reg(path_on_cloud, 3)
    ## make a html for user to click and view the detailed model result in PDF format
    knn_on_cloud = "model_result/KNN/APPL_knn_output.pdf"
    knn_url = storage.child(knn_on_cloud).get_url(None)

    ### xgboost
    xgboost_mse = xgboost_model(path_on_cloud)
    xgboost_on_cloud = "model_result/XGBoost/APPL_xgboost_output.pdf"
    xgboost_url = storage.child(xgboost_on_cloud).get_url(None)

    ## LSTM
    LSTM_mse = LSTM_model(path_on_cloud, 1)
    #LSTM_on_cloud = "model_result/LSTM/APPL_LSTM_output.pdf"
    #lstm_url = storage.child(LSTM_on_cloud).get_url(None)
    lstm_url = "https://firebasestorage.googleapis.com/v0/b/stock-price-ee1ca.appspot.com/o/model_result%2FLSTM%2FAAPL_LSTM_output.pdf?alt=media&token=138240c1-e399-437d-9663-88497fa6d041"

#xgboost_mse = xgboost_mse,
                           #xgboost_url = xgboost_url,
    return render_template("apple.html",
                           data = apple_data,
                           meta = meta_list,
                           stat = apple_stat,
                           feature = url_fea,
                           streaming = streaming,
                           lr_mse = lr_mse,
                           lr_url = lr_url,
                           knn_mse = knn_mse,
                           knn_url = knn_url,
                           xg_mse = xgboost_mse,
                           xg_url = xgboost_url,
                           LSTM_MSE = LSTM_mse,
                           lstm_url = lstm_url)
                           
    
   
    
    

    #return render_template("apple.html", var1 = m) 
###########################################################################################################################
# AMAZON CODE #
@app.route('/amazon')

def amazon():
    #f = request.files['file']
    #filename = secure_filename(f.filename)
    filename = 'templates/amazon.txt'
    file = open(filename,"r")
    #text = []
    m = file.read()
    m = [x.strip() for x in m.split('\n')]

    ## getting raw data
    ## output format: pd DataFrame
    path_on_cloud = "stock_csv/AMZN/raw_AMZN.csv"
    url = storage.child(path_on_cloud).get_url(None)
    amazon_data = pd.read_csv(url, header = 0)

    ## getting mata data
    ## output format: list
    meta_on_cloud = "meta/meta_data_AMZN.txt"
    url_meta = storage.child(meta_on_cloud).get_url(None)
    file = urllib.request.urlopen(url_meta).read().splitlines()
    meta_list = []
    for line in file:
        decoded_line = line.decode("utf-8")
        meta_list.append(decoded_line)

    ## getting summary statistics from pyspark
    ## output format: pd DataFrame
    stat_on_cloud = "features/summaryAmazon.csv"
    url_stat = storage.child(stat_on_cloud).get_url(None)
    amazon_stat = pd.read_csv((url_stat), usecols=['summary', 'open', 'high', 'low', 'close', 'volume'])
    amazon_stat = amazon_stat.values.tolist()
    ## getting feature extraction from pyspark
    ## output format: url link
    ## user can clink the link to redirect the txt file
    fea_on_cloud = "features/pca_Amazon.txt"
    url_fea = storage.child(fea_on_cloud).get_url(None)

    ## getting pyspark steaming
    ## Output fromat: string
    ##### Sample output should be look like this:
    ## The input csv is: apple_data.csv
    ## The data is streaming:
    ## True

    amazon_data.to_csv("amazon_data.csv", index=False)
    streaming = streaming_data("amazon_data.csv")


    ## linear regression
    amazon_df = spark.read.format('csv').option('header', True).schema(schema).load("amazon_data.csv")
    lr_mse = linear_regression(amazon_df)
    ## make a html for user to click and view the detailed model result in PDF format
    lr_on_cloud = "model_result/Linear_Regression/AMZN_lr_output.pdf"
    lr_url = storage.child(lr_on_cloud).get_url(None)

    ### KNN
    knn_mse = KNN_reg(path_on_cloud, 3)
    ## make a html for user to click and view the detailed model result in PDF format
    knn_on_cloud = "model_result/KNN/AMZN_knn_output.pdf"
    knn_url = storage.child(knn_on_cloud).get_url(None)

    ### xgboost
    xgboost_mse = xgboost_model(path_on_cloud)
    xgboost_on_cloud = "model_result/XGBoost/AMZN_xgboost_output.pdf"
    xgboost_url = storage.child(xgboost_on_cloud).get_url(None)

    ## LSTM
    LSTM_mse = LSTM_model(path_on_cloud, 1)
    LSTM_on_cloud = "model_result/LSTM/AMZN_LSTM_output.pdf"
    #lstm_url = storage.child(LSTM_on_cloud).get_url(None)
    lstm_url = "https://firebasestorage.googleapis.com/v0/b/stock-price-ee1ca.appspot.com/o/model_result%2FLSTM%2FAMZN_LSTM_output.pdf?alt=media&token=08e0a4c2-3ac1-4886-90c9-e92b5573bde5"
#xgboost_mse = xgboost_mse,
                           #xgboost_url = xgboost_url,
    return render_template("amazon.html",
                           data = amazon_data,
                           meta = meta_list,
                           stat = amazon_stat,
                           feature = url_fea,
                           streaming = streaming,
                           lr_mse = lr_mse,
                           lr_url = lr_url,
                           knn_mse = knn_mse,
                           knn_url = knn_url,
                           xg_mse = xgboost_mse,
                           xg_url = xgboost_url,
                           LSTM_MSE = LSTM_mse,
                           lstm_url = lstm_url)


###########################################################################################################################
# TESLA CODE #
@app.route('/tesla')

def tesla():
    #f = request.files['file']
    #filename = secure_filename(f.filename)
    filename = 'templates/tesla.txt'
    file = open(filename,"r")
    #text = []
    m = file.read()
    m = [x.strip() for x in m.split('\n')]

    ## getting raw data
    ## output format: pd DataFrame
    path_on_cloud = "stock_csv/TSLA/raw_TSLA.csv"
    url = storage.child(path_on_cloud).get_url(None)
    tesla_data = pd.read_csv(url, header = 0)

    ## getting mata data
    ## output format: list
    meta_on_cloud = "meta/meta_data_TSLA.txt"
    url_meta = storage.child(meta_on_cloud).get_url(None)
    file = urllib.request.urlopen(url_meta).read().splitlines()
    meta_list = []
    for line in file:
        decoded_line = line.decode("utf-8")
        meta_list.append(decoded_line)

    ## getting summary statistics from pyspark
    ## output format: pd DataFrame
    stat_on_cloud = "features/summaryTesla.csv"
    url_stat = storage.child(stat_on_cloud).get_url(None)
    tesla_stat = pd.read_csv((url_stat), usecols=['summary', 'open', 'high', 'low', 'close', 'volume'])
    tesla_stat = tesla_stat.values.tolist()
    ## getting feature extraction from pyspark
    ## output format: url link
    ## user can clink the link to redirect the txt file
    fea_on_cloud = "features/pca_Tesla.txt"
    url_fea = storage.child(fea_on_cloud).get_url(None)

    ## getting pyspark steaming
    ## Output fromat: string
    ##### Sample output should be look like this:
    ## The input csv is: apple_data.csv
    ## The data is streaming:
    ## True

    tesla_data.to_csv("tesla_data.csv", index=False)
    streaming = streaming_data("tesla_data.csv")


    ## linear regression
    tesla_df = spark.read.format('csv').option('header', True).schema(schema).load("tesla_data.csv")
    lr_mse = linear_regression(tesla_df)
    ## make a html for user to click and view the detailed model result in PDF format
    lr_on_cloud = "model_result/Linear_Regression/TSLA_lr_output.pdf"
    lr_url = storage.child(lr_on_cloud).get_url(None)

    ### KNN
    knn_mse = KNN_reg(path_on_cloud, 3)
    ## make a html for user to click and view the detailed model result in PDF format
    knn_on_cloud = "model_result/KNN/TSLA_knn_output.pdf"
    knn_url = storage.child(knn_on_cloud).get_url(None)

    ### xgboost
    xgboost_mse = xgboost_model(path_on_cloud)
    xgboost_on_cloud = "model_result/XGBoost/TSLA_xgboost_output.pdf"
    xgboost_url = storage.child(xgboost_on_cloud).get_url(None)

    ## LSTM
    LSTM_mse = LSTM_model(path_on_cloud, 1)
    LSTM_on_cloud = "model_result/LSTM/TSLA_LSTM_output.pdf"
    #lstm_url = storage.child(LSTM_on_cloud).get_url(None)
    lstm_url = "https://firebasestorage.googleapis.com/v0/b/stock-price-ee1ca.appspot.com/o/model_result%2FLSTM%2FTSLA_LSTM_output.pdf?alt=media&token=364c02c6-1f93-4ea8-9706-e559c9609d93"
#xgboost_mse = xgboost_mse,
                           #xgboost_url = xgboost_url,
    return render_template("tesla.html",
                           data = tesla_data,
                           meta = meta_list,
                           stat = tesla_stat,
                           feature = url_fea,
                           streaming = streaming,
                           lr_mse = lr_mse,
                           lr_url = lr_url,
                           knn_mse = knn_mse,
                           knn_url = knn_url,
                           xg_mse = xgboost_mse,
                           xg_url = xgboost_url,
                           LSTM_MSE = LSTM_mse,
                           lstm_url = lstm_url)


###########################################################################################################################
# GOOGLE CODE #
@app.route('/google')
def google():
    #f = request.files['file']
    #filename = secure_filename(f.filename)
    filename = 'templates/google.txt'
    file = open(filename,"r")
    #text = []
    m = file.read()
    m = [x.strip() for x in m.split('\n')]

    ## getting raw data
    ## output format: pd DataFrame
    path_on_cloud = "stock_csv/GOOD/raw_GOOD.csv"
    url = storage.child(path_on_cloud).get_url(None)
    google_data = pd.read_csv(url, header = 0)

    ## getting mata data
    ## output format: list
    meta_on_cloud = "meta/meta_data_GOOD.txt"
    url_meta = storage.child(meta_on_cloud).get_url(None)
    file = urllib.request.urlopen(url_meta).read().splitlines()
    meta_list = []
    for line in file:
        decoded_line = line.decode("utf-8")
        meta_list.append(decoded_line)

    ## getting summary statistics from pyspark
    ## output format: pd DataFrame
    stat_on_cloud = "features/summaryGoogle.csv"
    url_stat = storage.child(stat_on_cloud).get_url(None)
    google_stat = pd.read_csv((url_stat), usecols=['summary', 'open', 'high', 'low', 'close', 'volume'])
    google_stat = google_stat.values.tolist()
    ## getting feature extraction from pyspark
    ## output format: url link
    ## user can clink the link to redirect the txt file
    fea_on_cloud = "features/pca_Google.txt"
    url_fea = storage.child(fea_on_cloud).get_url(None)

    ## getting pyspark steaming
    ## Output fromat: string
    ##### Sample output should be look like this:
    ## The input csv is: apple_data.csv
    ## The data is streaming:
    ## True

    google_data.to_csv("google_data.csv", index=False)
    streaming = streaming_data("google_data.csv")


    ## linear regression
    google_df = spark.read.format('csv').option('header', True).schema(schema).load("google_data.csv")
    lr_mse = linear_regression(google_df)
    ## make a html for user to click and view the detailed model result in PDF format
    lr_on_cloud = "model_result/Linear_Regression/GOOD_lr_output.pdf"
    lr_url = storage.child(lr_on_cloud).get_url(None)

    ### KNN
    knn_mse = KNN_reg(path_on_cloud, 3)
    ## make a html for user to click and view the detailed model result in PDF format
    knn_on_cloud = "model_result/KNN/GOOD_knn_output.pdf"
    knn_url = storage.child(knn_on_cloud).get_url(None)

    ### xgboost
    xgboost_mse = xgboost_model(path_on_cloud)
    xgboost_on_cloud = "model_result/XGBoost/GOOD_xgboost_output.pdf"
    xgboost_url = storage.child(xgboost_on_cloud).get_url(None)

    ## LSTM
    LSTM_mse = LSTM_model(path_on_cloud, 1)
    LSTM_on_cloud = "model_result/LSTM/GOOD_LSTM_output.pdf"
    #lstm_url = storage.child(LSTM_on_cloud).get_url(None)
    lstm_url = "https://firebasestorage.googleapis.com/v0/b/stock-price-ee1ca.appspot.com/o/model_result%2FLSTM%2FAMZN_LSTM_output.pdf?alt=media&token=08e0a4c2-3ac1-4886-90c9-e92b5573bde5"
#xgboost_mse = xgboost_mse,
                           #xgboost_url = xgboost_url,
    return render_template("google.html",
                           data = google_data,
                           meta = meta_list,
                           stat = google_stat,
                           feature = url_fea,
                           streaming = streaming,
                           lr_mse = lr_mse,
                           lr_url = lr_url,
                           knn_mse = knn_mse,
                           knn_url = knn_url,
                           xg_mse = xgboost_mse,
                           xg_url = xgboost_url,
                           LSTM_MSE = LSTM_mse,
                           lstm_url = lstm_url)
###########################################################################################################################
# NVIDIA CODE #
@app.route('/nvidia')

def nvidia():
    #f = request.files['file']
    #filename = secure_filename(f.filename)
    filename = 'templates/nvidia.txt'
    file = open(filename,"r")
    #text = []
    m = file.read()
    m = [x.strip() for x in m.split('\n')]

    ## getting raw data
    ## output format: pd DataFrame
    path_on_cloud = "stock_csv/NVDA/raw_NVDA.csv"
    url = storage.child(path_on_cloud).get_url(None)
    nvidia_data = pd.read_csv(url, header = 0)

    ## getting mata data
    ## output format: list
    meta_on_cloud = "meta/meta_data_NVDA.txt"
    url_meta = storage.child(meta_on_cloud).get_url(None)
    file = urllib.request.urlopen(url_meta).read().splitlines()
    meta_list = []
    for line in file:
        decoded_line = line.decode("utf-8")
        meta_list.append(decoded_line)

    ## getting summary statistics from pyspark
    ## output format: pd DataFrame
    stat_on_cloud = "features/summaryNVIDIA.csv"
    url_stat = storage.child(stat_on_cloud).get_url(None)
    nvidia_stat = pd.read_csv((url_stat), usecols=['summary', 'open', 'high', 'low', 'close', 'volume'])
    nvidia_stat = nvidia_stat.values.tolist()
    ## getting feature extraction from pyspark
    ## output format: url link
    ## user can clink the link to redirect the txt file
    fea_on_cloud = "features/pca_NVIDIA.txt"
    url_fea = storage.child(fea_on_cloud).get_url(None)

    ## getting pyspark steaming
    ## Output fromat: string
    ##### Sample output should be look like this:
    ## The input csv is: apple_data.csv
    ## The data is streaming:
    ## True

    nvidia_data.to_csv("nvidia_data.csv", index=False)
    streaming = streaming_data("nvidia_data.csv")


    ## linear regression
    nvidia_df = spark.read.format('csv').option('header', True).schema(schema).load("nvidia_data.csv")
    lr_mse = linear_regression(nvidia_df)
    ## make a html for user to click and view the detailed model result in PDF format
    lr_on_cloud = "model_result/Linear_Regression/NVDA_lr_output.pdf"
    lr_url = storage.child(lr_on_cloud).get_url(None)

    ### KNN
    knn_mse = KNN_reg(path_on_cloud, 3)
    ## make a html for user to click and view the detailed model result in PDF format
    knn_on_cloud = "model_result/KNN/NVDA_knn_output.pdf"
    knn_url = storage.child(knn_on_cloud).get_url(None)

    ### xgboost
    xgboost_mse = xgboost_model(path_on_cloud)
    xgboost_on_cloud = "model_result/XGBoost/NVDA_xgboost_output.pdf"
    xgboost_url = storage.child(xgboost_on_cloud).get_url(None)

    ## LSTM
    LSTM_mse = LSTM_model(path_on_cloud, 1)
    LSTM_on_cloud = "model_result/LSTM/NVDA_LSTM_output.pdf"
    #lstm_url = storage.child(LSTM_on_cloud).get_url(None)
    lstm_url = "https://firebasestorage.googleapis.com/v0/b/stock-price-ee1ca.appspot.com/o/model_result%2FLSTM%2FNVDA_LSTM_output.pdf?alt=media&token=097e15f7-d066-4fd5-83e8-5d859f981b6a"
#xgboost_mse = xgboost_mse,
                           #xgboost_url = xgboost_url,
    return render_template("nvidia.html",
                           data = nvidia_data,
                           meta = meta_list,
                           stat = nvidia_stat,
                           feature = url_fea,
                           streaming = streaming,
                           lr_mse = lr_mse,
                           lr_url = lr_url,
                           knn_mse = knn_mse,
                           knn_url = knn_url,
                           xg_mse = xgboost_mse,
                           xg_url = xgboost_url,
                           LSTM_MSE = LSTM_mse,
                           lstm_url = lstm_url)
   
if __name__ == '__main__':  
    app.run(debug = True)  






