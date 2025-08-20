// Function to send user message to Flask backend
async function sendMessage() {
    let inputField = document.getElementById("user-input");
    let chatLog = document.getElementById("chat-log");
    let userMessage = inputField.value.trim();

    if (!userMessage) return;

    // Show user message
    chatLog.innerHTML += `<p><b>You:</b> ${userMessage}</p>`;
    inputField.value = "";

    // Send message to backend
    try {
        let response = await fetch("/get_response", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: userMessage })
        });

        let data = await response.json();

        // Show bot reply
        chatLog.innerHTML += `<p><b>Bot:</b> ${data.response}</p>`;
        chatLog.scrollTop = chatLog.scrollHeight;  // Auto-scroll

        // Play audio if provided
        if (data.audio) {
            let audio = new Audio(data.audio);
            audio.play();
        }
    } catch (error) {
        console.error("Error:", error);
        chatLog.innerHTML += `<p style="color:red;"><b>Bot:</b> Oops! Something went wrong.</p>`;
    }
}

// Allow pressing Enter to send message
document.getElementById("user-input").addEventListener("keypress", function(e) {
    if (e.key === "Enter") {
        sendMessage();
    }
});
