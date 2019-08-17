import time
import pika
import json
from pika import exceptions
from src.predictor.image.prepDataIO import prep_data
from src.predictor.predict import predict
import os

# Queueing declaration stuff here.

queue_host = os.getenv('AMQP_HOST', 'localhost')
queue_user = os.getenv('AMQP_USER', 'guest')
queue_password = os.getenv('AMQP_PASSWORD', 'guest')

connection = None
for i in range(0,10):
    while True:
        try:
            print("Attempting connection to host: " + queue_host)
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=queue_host,
                                                                           credentials=pika.credentials.PlainCredentials(queue_user, queue_password)))
        except exceptions.AMQPConnectionError:
            print("Retry in 1 second...")
            time.sleep(1)
            continue
        break


channel = connection.channel()

channel.queue_declare(queue='clj->py')

nii_input_path = os.getenv('NII_INPUT_PATH',
                          "/Users/pmartyn/Documents/College_Work/Thesis/Thesis-Frontend/thesis-frontend/efs/")

nii_base_path = os.getenv('OUTPUT_FILE_PATH',
                          "/Users/pmartyn/PycharmProjects/Thesis/tmp/")


# Setup the callback to run the predictor and put the results on the return queue.
def message_delivery(ch, method, properties, body):

    print("Request received!")
    print("Reply-to queue: " + str(properties.reply_to))

    # Get the correlation id from the request.
    # This needs to be returned to the server
    correlation_id_str = str(properties.correlation_id)
    print("Request correlation id: " + correlation_id_str)

    # Decode the request
    request = json.loads(body.decode("utf-8"))

    nii_filename = request["filename"]
    print("Request nii_filename: " + nii_filename)

    # Create the unique filepath using the correlation id
    nii_output_path = nii_base_path + correlation_id_str + '/'
    # Create the path to the file. Will be on EFS on AWS.
    nii_file_path = nii_input_path + correlation_id_str + ".nii"

    # Create directory if necessary
    if not os.path.exists(nii_output_path):
        os.makedirs(nii_output_path)

    if nii_filename and os.path.exists(nii_file_path):

        prep_data(nii_file_path, nii_output_path)

        try:
            # Predict and get the results
            response_data = predict(nii_output_path)
            print("Return the response: " + str(response_data))
            # Remove the request file, confidential image data cannot be left on the server.
            os.remove(nii_file_path)
            # Publish back to the frontend the result data.

            channel.basic_publish(exchange='',
                                  routing_key=properties.reply_to,
                                  properties=pika.BasicProperties(correlation_id=properties.correlation_id),
                                  body=json.dumps(response_data))

        except ValueError as e:
            print("Cannot do prediction: ", e)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(
    queue='clj->py',
    on_message_callback=message_delivery,
    auto_ack=True)


print('Python predictor waiting for messages. To exit press CTRL+C')
channel.start_consuming()
