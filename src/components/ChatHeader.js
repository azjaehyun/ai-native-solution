import React from "react";
import { AppBar, Toolbar, IconButton, Typography, Select, MenuItem } from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";

const ChatHeader = ({ isSidebarOpen, toggleSidebar, selectedModel, handleModelChange }) => (
  <AppBar position="static" sx={{ bgcolor: "#333" }}>
    <Toolbar>
      {!isSidebarOpen && (
        <IconButton edge="start" color="inherit" onClick={toggleSidebar} sx={{ mr: 2 }}>
          <MenuIcon />
        </IconButton>
      )}
      <Typography variant="h6" sx={{ flexGrow: 1 }}>
        AWS BedRock PoC
      </Typography>
      <Select value={selectedModel} onChange={handleModelChange} sx={{ bgcolor: "#444", color: "#fff" }}>
        <MenuItem value="ChatGPT 4.0">ChatGPT 4.0</MenuItem>
        <MenuItem value="ChatGPT 3.5">ChatGPT 3.5</MenuItem>
      </Select>
    </Toolbar>
  </AppBar>
);

export default ChatHeader;
