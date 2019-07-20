import pika
import json
from src.Predictor.image.prepDataIO import prep_data
from src.Predictor.predict import predict
import os

# Queueing declaration stuff here.
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

channel.queue_declare(queue='py->clj')
channel.queue_declare(queue='clj->py')

nii_base_path = os.getenv('NII_PATH',
                          "/Users/pmartyn/Documents/College_Work/Thesis/Thesis-Frontend/thesis-frontend/resources/public/")

# Setup the callback to run the predictor and put the results on the return queue.
def json_callback(ch, method, properties, body):

    request = json.loads(body.decode("utf-8"))
    print("Request from frontend:  " + str(request))
    nii_filename = request["filename"]
    nii_path = nii_base_path + nii_filename

    if nii_filename and os.path.exists(nii_path):

        prep_data(nii_path)

        try:
            # Predict and get the results
            response_data = predict()
            print("Return the response: " + str(response_data))
            # Remove the request file, confidential image data cannot be left on the server.
            os.remove(nii_path)
            # Publish back to the frontend the result data.
            channel.basic_publish(exchange='', routing_key='py->clj', body=json.dumps(response_data))

        except ValueError as e:
            print("Cannot do prediction: ", e)


channel.basic_consume(
    queue='clj->py',
    on_message_callback=json_callback,
    auto_ack=True)


print('Python predictor waiting for messages. To exit press CTRL+C')
channel.start_consuming()
