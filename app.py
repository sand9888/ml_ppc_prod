#! /usr/bin/python3

from sanic import Sanic
from sanic import response
from sanic.response import json, file, text
from sanic.views import HTTPMethodView
from os.path import join, dirname
import os
import smtplib
import json as j 
from xgboost_cpc import modelfit
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier


app = Sanic()
app.static('/forsolving.com', './forsolving.com')


@app.route("/")
@app.route("/index.html")
def index(request):
    response = file(join(dirname(__file__),'forsolving.com/index.html'))
    return response


@app.route("/mail.html", methods=['POST'])
def mail(request):
    email = request.body.decode()
    print(email)
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("YOUR EMAIL ADDRESS", "YOUR PASSWORD")
    msg = "YOUR MESSAGE!"
    if email:
        server.sendmail("YOUR EMAIL ADDRESS", email, msg)
    server.quit()
    return text('Спасибо. В ближайшее время мы свяжемся с Вами')


@app.route("/team.html")
def team(request):
    response = file(join(dirname(__file__),'forsolving.com/team.html'))
    return response


@app.route("/tools.html")
def team(request):
    response = file(join(dirname(__file__),'forsolving.com/tools.html'))
    return response


@app.route("/contacts.html")
def contacts(request):
    response = file(join(dirname(__file__),'forsolving.com/contacts.html'))
    return response

class SimpleView(HTTPMethodView):
    def get(self, request):
        response = file(join(dirname(__file__),'forsolving.com/ppc.html'))
        return response

    def post(self, request):

        file = open('data.csv', 'w')
        file.write(request.body.decode())
        file.close()
        try:
            train = pd.read_csv('data.csv', header=0, encoding='latin1')
            train.drop(['id', 'Unnamed: 0', 'index', 'conv_rate'], axis=1, inplace=True)
            train['bounce_rate'] = train['bounce_rate'].astype(float)


            x = train.values
            x_scaled = preprocessing.MinMaxScaler().fit_transform(x)
            df = pd.DataFrame(x_scaled, columns=train.columns)
            train, test = train_test_split(df, test_size = 0.4)

            target = 'with_conversion'
            IDcol = 'id'

            xgb4 = XGBClassifier(
                    learning_rate=0.01,
                    n_estimators=5000,
                    max_depth=2,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.85,
                    colsample_bytree=0.9,
                    reg_alpha=0.005,
                    objective='binary:logistic',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)

            predictors = [x for x in train.columns if x not in [target, IDcol]]
                
            data = modelfit(xgb4, train, test, predictors)
            #modelfit(xgb4, train, test, predictors)
            os.remove('data.csv')
            return response.json(j.dumps(data))
        
        except:
            return response.json(j.dumps('Something went wrong'),status=500)

app.add_route(SimpleView.as_view(), '/ppc.html')

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )
