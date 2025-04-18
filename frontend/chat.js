// Firebase Firestore
const db = firebase.firestore();

// DOM Elements
const logoutBtn = document.getElementById('logoutBtn');
const chatForm = document.getElementById('chatForm');
const promptInput = document.getElementById('promptInput');
const chatMessages = document.getElementById('chatMessages');
const welcomeState = document.getElementById('welcomeState');
const newChatBtn = document.getElementById('newChatBtn');
const chatHistory = document.getElementById('chatHistory');

// Chat State
let currentChatId = null;
let chats = [];

// Auto-resize input
promptInput.addEventListener('input', function () {
  this.style.height = 'auto';
  this.style.height = (this.scrollHeight) + 'px';
});

// New Chat Button
newChatBtn.addEventListener('click', () => {
  createNewChat();
  showWelcomeState();
  promptInput.value = '';
  promptInput.style.height = 'auto';
});

// Create New Chat
function createNewChat() {
  const chatId = Date.now().toString();
  const newChat = {
    id: chatId,
    title: 'New Chat',
    messages: []
  };
  chats.push(newChat);
  currentChatId = chatId;

  chatMessages.innerHTML = '';
  addChatToHistory(newChat);
  updateActiveChatState(chatId);
}

// Show/Hide Welcome State
function showWelcomeState() {
  welcomeState.style.display = 'flex';
  chatMessages.style.display = 'none';
}

function hideWelcomeState() {
  welcomeState.style.display = 'none';
  chatMessages.style.display = 'block';
}

// Add Chat to Sidebar
function addChatToHistory(chat) {
  const chatElement = document.createElement('div');
  chatElement.className = 'chat-item';
  chatElement.setAttribute('data-chat-id', chat.id);
  chatElement.innerHTML = `<span>${chat.title}</span>`;

  chatElement.addEventListener('click', () => loadChat(chat.id));
  chatHistory.insertBefore(chatElement, chatHistory.firstChild);
}

// Load Chat
function loadChat(chatId) {
  const chat = chats.find(c => c.id === chatId);
  if (!chat) return;

  currentChatId = chatId;
  chatMessages.innerHTML = '';

  if (chat.messages.length === 0) {
    showWelcomeState();
  } else {
    hideWelcomeState();
    chat.messages.forEach(msg => {
      addMessage(msg.role, msg.content);
    });
  }

  updateActiveChatState(chatId);
}

// Update Sidebar Active State
function updateActiveChatState(chatId) {
  document.querySelectorAll('.chat-item').forEach(item => {
    item.classList.remove('active');
    if (item.getAttribute('data-chat-id') === chatId) {
      item.classList.add('active');
    }
  });
}

// Add Message to Chat
function addMessage(role, content) {
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.innerHTML = role === 'user' ? 'U' : 'AI';

  const messageContent = document.createElement('div');
  messageContent.className = 'message-content';
  messageContent.textContent = content;

  messageDiv.appendChild(avatar);
  messageDiv.appendChild(messageContent);
  chatMessages.appendChild(messageDiv);

  chatMessages.scrollTop = chatMessages.scrollHeight;

  return messageDiv;
}

// Update Chat Title
function updateChatTitle(chatId, newTitle) {
  const chatElement = document.querySelector(`.chat-item[data-chat-id="${chatId}"]`);
  if (chatElement) {
    chatElement.querySelector('span').textContent = newTitle;
  }
}

// Logout
logoutBtn.addEventListener('click', async () => {
  try {
    await firebase.auth().signOut();
    window.location.href = "login.html";  // Redirect to login page after logout
  } catch (error) {
    console.error("Error logging out: ", error);
    alert('Error logging out: ' + error.message);
  }
});
