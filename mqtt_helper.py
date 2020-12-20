import paho.mqtt.client as mqtt
import json

def get_topic_config(opt):
    mqtt_topic = opt.broker_topic
    broker_url = opt.broker_url
    broker_port = int(opt.broker_port)
    username = opt.broker_username
    password = opt.broker_password

    return {
		"topic": mqtt_topic, 
		"broker_url": broker_url, 
		"broker_port": broker_port, 
		"username": username,
		"password": password,
	}

def mqtt_topic(opt):
	mylist = []
	mycount = []
    
	config = get_topic_config(opt)
	
	client = mqtt.Client()
	client.username_pw_set(config['username'])
	client.connect(config["broker_url"], config["broker_port"])
	client.loop_start()
	return client

def publish_results(opt, client, detections_classes_numbers):
    config = get_topic_config(opt)
    MQTT_MSG = json.dumps(detections_classes_numbers)
    client.publish(config['topic'], MQTT_MSG)
