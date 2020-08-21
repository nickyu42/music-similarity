import sys
import os
import pathlib
import pickle

from flask import Flask, Blueprint, jsonify, render_template, request, current_app

from analysis.classification import MusicSimModel

main_page = Blueprint('main', __name__)


@main_page.route('/')
def index():
    return render_template('index.html')


@main_page.route('/song', methods=['POST'])
def song():
    file = request.files['song']
    model: MusicSimModel = current_app.config['MODEL']

    result = {
        'label': 'foo',
        'probability': 100.0
    }

    return jsonify(result)


def load_model():
    if 'MODEL_PATH' not in os.environ:
        raise ValueError('Environment variable MODEL_PATH not set')

    model_path = pathlib.Path(os.environ.get('MODEL_PATH'))

    if not model_path.is_file():
        raise ValueError('MODEL_PATH does not exist or is not a file.')

    with model_path.open('rb') as file:
        return pickle.load(file)


def create_app():
    app = Flask(__name__)
    app.register_blueprint(main_page)

    if os.environ.get('FLASK_ENV') == 'production':
        app.config['MODEL'] = load_model()

    return app
