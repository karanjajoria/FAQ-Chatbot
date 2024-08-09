document.addEventListener('DOMContentLoaded', function() {
    const sendBtn = document.getElementById('sendBtn');
    const userInput = document.getElementById('userInput');
    const chatMessages = document.getElementById('chatMessages');

    sendBtn.addEventListener('click', function() {
        const userText = userInput.value;
        if (userText.trim() !== '') {
            addMessage(userText, 'user');
            // Simulate bot response
            setTimeout(() => {
                addMessage('This is a bot response.', 'bot');
            }, 1000);
        }
        userInput.value = '';
    });

    function addMessage(text, sender) {
        const message = document.createElement('div');
        message.classList.add('message', sender);
        message.textContent = text;
        chatMessages.appendChild(message);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && userInput.value.trim() !== '') {
            sendBtn.click();
        }
    });
});
