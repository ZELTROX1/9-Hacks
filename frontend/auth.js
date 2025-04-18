const firebaseConfig = {
  apiKey: "AIzaSyC8LR_Xo3-GWdARHt_P4pjSjTaiml-_mvI",
  authDomain: "ai-track.firebaseapp.com",
  projectId: "ai-track",
  storageBucket: "ai-track.appspot.com",
  messagingSenderId: "962919661359",
  appId: "1:962919661359:web:9095f45fb894d54912ebda",
  measurementId: "G-K0WWJMVQ2G"
};

firebase.initializeApp(firebaseConfig);
const auth = firebase.auth();

function login(email, password) {
  auth.signInWithEmailAndPassword(email, password)
    .then(() => {
      showError(""); 
      window.location.href = "chat.html";
    })
    .catch((error) => {
      console.error('Login error:', error);
      const errorMsg = mapFirebaseError(error.code);
      showError(errorMsg);
    });
}

function signup(email, password) {
  auth.createUserWithEmailAndPassword(email, password)
    .then(() => {
      showError(""); 
      window.location.href = "chat.html";
    })
    .catch((error) => {
      console.error('Signup error:', error);
      const errorMsg = mapFirebaseError(error.code);
      showError(errorMsg);
    });
}

function logout() {
  auth.signOut()
    .then(() => {
      window.location.href = "index.html";
    })
    .catch((error) => {
      console.error('Logout error:', error);
    });
}

function showError(message) {
  const errorDiv = document.getElementById("errorMessage");
  if (errorDiv) {
    errorDiv.textContent = message;
    if (message) {
      errorDiv.classList.add("show");
    } else {
      errorDiv.classList.remove("show");
    }
  }
}

function mapFirebaseError(code) {
  switch (code) {
    case 'auth/user-not-found':
      return "No account found with this email.";
    case 'auth/wrong-password':
      return "Incorrect password.";
    case 'auth/email-already-in-use':
      return "Email already in use.";
    case 'auth/invalid-email':
      return "Invalid email address.";
    case 'auth/weak-password':
      return "Password should be at least 6 characters.";
    default:
      return "Something went wrong. Please try again.";
  }
}
