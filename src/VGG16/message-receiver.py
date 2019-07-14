import pika
import json
from src.VGG16.image.prepDataIO import prep_data
from src.VGG16.predict import predict
import os

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()


channel.queue_declare(queue='py->clj')
channel.queue_declare(queue='clj->py')

nii_base_path = (os.environ['NII_PATH'] or
                 "/Users/pmartyn/Documents/College_Work/Thesis/Thesis-Frontend/thesis-frontend/resources/public/")


def json_callback(ch, method, properties, body):
    request = json.loads(body.decode("utf-8"))
    print("python got from clojure: " + str(request))
    nii_filename = request["filename"]
    nii_path = nii_base_path + nii_filename

    if nii_filename and os.path.exists(nii_path):

        prep_data(nii_path)

        try:
            response_data = predict()
            print("python sez to clojure: " + str(response_data))
            os.remove(nii_path)
            channel.basic_publish(exchange='', routing_key='jsonpy2clj', body=json.dumps(response_data))
        except ValueError as e:
            print("Cannot do prediction: ", e)


channel.basic_consume(
    queue='clj->py',
    on_message_callback=json_callback,
    auto_ack=True)


print('Python predictor waiting for messages. To exit press CTRL+C')
channel.start_consuming()
