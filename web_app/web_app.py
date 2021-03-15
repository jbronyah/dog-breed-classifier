# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 16:07:32 2021

@author: BronyahJ
"""
import json
import plotly
import pandas as pd
from flask import Flask, request, jsonify, render_template
from plotly.graph_objs import Bar

app = Flask(__name__)

@app.route('/')
#@app.route('/index')
@app.route('/go')

def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    #classification_labels = model.predict([query])[0]
    #classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        #classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)
    

if __name__ == '__main__':
    main()