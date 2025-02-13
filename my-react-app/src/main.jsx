// main.jsx (or index.jsx, depending on your setup)
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';  // Import your App component
import './App.css';  // Import global styles (if you have)

const root = ReactDOM.createRoot(document.getElementById('root'));  // Ensure 'root' is in your HTML
root.render(
  <React.StrictMode>
    <App />  {/* Render the App component inside StrictMode for extra checks */}
  </React.StrictMode>
);
