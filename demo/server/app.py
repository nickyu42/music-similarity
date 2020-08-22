import sys
import os
import pathlib
import pickle
import uuid
from multiprocessing import Process, Queue
from cachetools import TTLCache

from flask import Flask, Blueprint, jsonify, render_template, request, current_app, abort

from analysis.classification import MusicSimModel

CACHE_SIZE = 1024
CACHE_TTL = 120

main_page = Blueprint('main', __name__)


@main_page.route('/')
def index():
    return render_template('index.html')


@main_page.route('/status', methods=['POST'])
def status():
    if not request.is_json or 'id' not in request.json:
        abort(400, 'The request should be a of type "application/json" and should contain an "id" key')

    client_id = request.json['id']
    results_cache = current_app.config['result_cache']

    if client_id not in results_cache:
        return jsonify(done=False)

    success, label, probability = results_cache[client_id]

    return jsonify(done=True, success=success, label=label, probability=probability)


@main_page.route('/song', methods=['POST'])
def song():
    song = request.files['song']

    client_id = uuid.uuid4().int
    queue: Queue = current_app.config['in_queue']

    # XXX: Fix file queuing
    # file handles cannot be serialized properly, so maybe send the file as a bytearray?
    # another option would be to store the file temporarily on the filesystem
    queue.put((client_id, song))

    return jsonify(id=client_id), 202


def load_model():
    if 'MODEL_PATH' not in os.environ:
        raise ValueError('Environment variable MODEL_PATH not set')

    model_path = pathlib.Path(os.environ.get('MODEL_PATH'))

    if not model_path.is_file():
        raise ValueError('MODEL_PATH does not exist or is not a file.')

    with model_path.open('rb') as file:
        return pickle.load(file)


def job_handler(model: MusicSimModel, in_queue: Queue, out_queue: Queue):
    while True:
        client_id, song = in_queue.get()

        # TODO: add model prediction

        out_queue.put((client_id, {'success': True, 'label': 'shounen', probability: 100.0}))


def init_production(app: Flask):
    # TODO: add proper model
    # model = load_model()
    model = None

    in_queue = Queue()
    out_queue = Queue()

    # Add reference to input queue in the app's config
    app.config['in_queue'] = in_queue

    # Start background song predictor
    p = Process(target=job_handler, args=(model, in_queue, out_queue), daemon=True)
    p.start()


def create_app():
    app = Flask(__name__)
    app.register_blueprint(main_page)

    app.config['results_cache'] = TTLCache(CACHE_SIZE, CACHE_TTL)

    if os.environ.get('FLASK_ENV') == 'production':
        init_production(app)

    return app
