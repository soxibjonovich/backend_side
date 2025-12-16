from api.run import app

if __name__ == '__main__':
	import uvicorn
	uvicorn.run(
	app, 
	host="0.0.0.0", 
	port=8000,
	ssl_certfile="cert.pem",
	ssl_keyfile="key.pem"
	)