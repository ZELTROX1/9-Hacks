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

// Send Prompt to Flask API
async function sendPromptToAPI(prompt) {
  const apiUrl = 'http://localhost:5000/api/chat';
  const response = await fetch(apiUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ prompt: prompt })
  });

  if (!response.ok) {
    throw new Error('Failed to fetch AI response');
  }

  const data = await response.json();
  return data.response || "No response received.";
}

// Handle Form Submit
chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  hideWelcomeState();

  const currentUser = firebase.auth().currentUser;
  if (!currentUser) {
    window.location.href = "login.html";
    return;
  }

  addMessage('user', prompt);

  const currentChat = chats.find(c => c.id === currentChatId);
  if (currentChat) {
    currentChat.messages.push({ role: 'user', content: prompt });

    if (currentChat.messages.length === 1) {
      currentChat.title = prompt.slice(0, 30) + (prompt.length > 30 ? '...' : '');
      updateChatTitle(currentChatId, currentChat.title);
    }
  }

  promptInput.value = '';
  promptInput.style.height = 'auto';

  const loadingMessage = addMessage('assistant', 'Thinking...');

  try {
    const apiResponse = await sendPromptToAPI(prompt);

    const cleanedResponse = apiResponse.replace(/\n+/g, '\n').trim();

    loadingMessage.querySelector('.message-content').textContent = cleanedResponse;

    if (currentChat) {
      currentChat.messages.push({ role: 'assistant', content: cleanedResponse });

      // Save chat to Firestore
      await db.collection('users')
        .doc(currentUser.uid)
        .collection('chats')
        .doc(currentChatId)
        .set(currentChat);
    }
  } catch (error) {
    loadingMessage.querySelector('.message-content').textContent = 'Error: ' + error.message;
  }
});

// Load User's Existing Chats
async function loadUserChats() {
  const currentUser = firebase.auth().currentUser;
  if (!currentUser) {
    window.location.href = "login.html";
    return;
  }

  const chatsSnapshot = await db.collection('users')
    .doc(currentUser.uid)
    .collection('chats')
    .get();

  chats = chatsSnapshot.docs.map(doc => doc.data());

  chats.forEach(chat => {
    addChatToHistory(chat);
  });

  if (chats.length > 0) {
    loadChat(chats[0].id); // Load first chat automatically
  } else {
    showWelcomeState();
  }
}

// On Firebase Ready
firebase.auth().onAuthStateChanged(user => {
  if (user) {
    loadUserChats();
  } else {
    window.location.href = "login.html";
  }
});

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
