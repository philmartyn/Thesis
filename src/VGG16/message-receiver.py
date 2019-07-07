import pika
import json
from Data import prepData
from src.VGG16 import predict
import os
#
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()


channel.queue_declare(queue='jsonpy2clj')
channel.queue_declare(queue='jsonclj2py')


def json_callback(ch, method, properties, body):
    request = json.loads(body.decode("utf-8"))
    print("python got from clojure: " + str(request))
    filename = request["filename"]
    nii_path = "/Users/pmartyn/Documents/College_Work/Thesis/Thesis-Frontend/thesis-frontend/resources/public/" + filename

    if filename and os.path.exists(nii_path):

        prepData.prep_data(filename)

        response_data = predict.predict()
        print("python sez to clojure: " + str(response_data))

        os.remove(nii_path)

        channel.basic_publish(exchange='', routing_key='jsonpy2clj', body=json.dumps(response_data))


channel.basic_consume(
    queue='jsonclj2py',
    on_message_callback=json_callback,
    auto_ack=True)


print('Python Waiting for messages. To exit press CTRL+C')
channel.start_consuming()
