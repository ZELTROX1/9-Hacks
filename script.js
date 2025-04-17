document.addEventListener("DOMContentLoaded", function () {
    // Directly show main app (no login/signup)
    document.getElementById("mainApp").classList.remove("hidden");

    const chatForm = document.getElementById("chatForm");
    const promptInput = document.getElementById("promptInput");
    const chatMessages = document.getElementById("chatMessages");
    const newChatBtn = document.querySelector(".new-chat-btn");
    const logoutBtn = document.getElementById("logoutBtn");

    chatForm.addEventListener("submit", function (e) {
        e.preventDefault();
        const userMessage = promptInput.value.trim();
        if (userMessage === "") return;

        appendMessage("You", userMessage);
        promptInput.value = "";

        // Simulated AI response (replace with actual API call if needed)
        setTimeout(() => {
            const aiResponse = "This is a simulated AI response to: " + userMessage;
            appendMessage("AI", aiResponse);
        }, 500);
    });

    function appendMessage(sender, message) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message");
        messageDiv.innerHTML = `<strong>${sender}:</strong> ${message}`;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    newChatBtn.addEventListener("click", () => {
        chatMessages.innerHTML = "";
    });

    logoutBtn.addEventListener("click", () => {
        // If login was ever re-added, you could clear login status here
        // localStorage.removeItem("isLoggedIn");
        alert("Logged out (no login system active).");
        chatMessages.innerHTML = "";
    });
});
