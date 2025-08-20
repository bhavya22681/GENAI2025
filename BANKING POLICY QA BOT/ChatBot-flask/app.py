from flask import Flask, render_template, request, jsonify
from bankbot import get_response   # ✅ import directly from package

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    bot_reply = get_response(user_input)
    return jsonify({"response": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)



