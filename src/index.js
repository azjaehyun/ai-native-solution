import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import './index.css';
import SignIn from './components/sign-in-side/SignInSide';
import App from './App';
import ProtectedRoute from './components/ProtectedRoute';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <Router>
    <Routes>
      <Route path="/" element={<SignIn />} />
      <Route
        path="/chat"
        element={
          <ProtectedRoute>
            <App />
          </ProtectedRoute>
        }
      />
      {/* Redirect to / if no match is found */}
      <Route path="*" element={<Navigate to="/" />} />
    </Routes>
  </Router>
);