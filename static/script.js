
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatArea = document.getElementById('chat-area');
    const micBtn = document.getElementById('mic-btn');
    const sendBtn = document.getElementById('send-btn');
    const convoList = document.getElementById('conversation-list');
    const newConvoBtn = document.getElementById('new-convo-btn');
    const connectionStatus = document.getElementById('connectionStatus');

    // API Configuration
    const API_BASE_URL = 'http://localhost:8000';
    const WEBSOCKET_URL = 'ws://localhost:8000';
    
    // Voice conversation state
    let isListening = false;
    let isSpeaking = false;
    let continuousMode = false;
    let recognition;
    let currentUtterance = null;

    // Conversation state
    let conversations = [];
    let currentConvoIdx = 0;
    let currentSessionId = generateSessionId();
    let websocket = null;
    let isConnected = false;
    let isWaitingForResponse = false;
    
    // Logout function
    function logout() {
      // Stop all ongoing processes
      stopSpeaking();
      stopListening();
      continuousMode = false;
      
      // Close WebSocket connection
      if (websocket) {
        websocket.close();
      }
      
      // Clear any stored session data
      conversations = [];
      currentConvoIdx = 0;
      currentSessionId = generateSessionId();
      
      // Make logout request to server
      fetch('/logout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      }).then(() => {
        // Redirect to login page
        window.location.href = '/static/login.html';
      }).catch((error) => {
        console.error('Logout error:', error);
        // Redirect anyway
        window.location.href = '/static/login.html';
      });
    }
    
    // Initialize conversations
    function initializeConversations() {
      conversations = [{
        id: currentSessionId,
        started: new Date().toISOString(),
        messages: [
          { sender: 'ai', text: "Hello! I'm Budger, your AI assistant from Cogent Infotech Corporation. How can I help you today?", time: getCurrentTime() }
        ]
      }];
      currentConvoIdx = 0;
    }

    function generateSessionId() {
      return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
    }

    function getCurrentTime() {
      const now = new Date();
      let hours = now.getHours();
      let minutes = now.getMinutes();
      const ampm = hours >= 12 ? 'PM' : 'AM';
      hours = hours % 12;
      hours = hours ? hours : 12;
      minutes = minutes < 10 ? '0' + minutes : minutes;
      return `${hours}:${minutes} ${ampm}`;
    }

    function updateConnectionStatus(status) {
      connectionStatus.className = `connection-status ${status}`;
      switch(status) {
        case 'connected':
          connectionStatus.textContent = 'Connected';
          isConnected = true;
          break;
        case 'connecting':
          connectionStatus.textContent = 'Connecting...';
          isConnected = false;
          break;
        case 'disconnected':
          connectionStatus.textContent = 'Disconnected';
          isConnected = false;
          break;
      }
    }

    // WebSocket Connection
    function connectWebSocket() {
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        return;
      }

      updateConnectionStatus('connecting');
      
      try {
        websocket = new WebSocket(`${WEBSOCKET_URL}/ws/${currentSessionId}`);
        
        websocket.onopen = function(event) {
          console.log('WebSocket connected');
          updateConnectionStatus('connected');
        };
        
        websocket.onmessage = function(event) {
          try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };
        
        websocket.onclose = function(event) {
          console.log('WebSocket disconnected');
          updateConnectionStatus('disconnected');
          
          // Attempt to reconnect after 3 seconds
          setTimeout(() => {
            if (!isConnected) {
              connectWebSocket();
            }
          }, 3000);
        };
        
        websocket.onerror = function(error) {
          console.error('WebSocket error:', error);
          updateConnectionStatus('disconnected');
        };
      } catch (error) {
        console.error('Error creating WebSocket:', error);
        updateConnectionStatus('disconnected');
      }
    }

    function handleWebSocketMessage(data) {
      const currentConvo = conversations[currentConvoIdx];
      const lastMessage = currentConvo.messages[currentConvo.messages.length - 1];
      
      if (data.type === 'partial') {
        // Update the streaming response
        if (lastMessage && lastMessage.sender === 'ai' && lastMessage.isStreaming) {
          lastMessage.text = data.full_response;
          renderChat(currentConvoIdx);
        }
      } else if (data.type === 'complete') {
        // Finalize the response
        if (lastMessage && lastMessage.sender === 'ai' && lastMessage.isStreaming) {
          lastMessage.text = data.content;
          lastMessage.isStreaming = false;
          lastMessage.sources = data.sources;
          renderChat(currentConvoIdx);
          renderSidebar();
          
          // Speak the response
          speakText(data.content);
        }
        isWaitingForResponse = false;
        sendBtn.disabled = false;
      } else if (data.type === 'error') {
        // Handle error
        if (lastMessage && lastMessage.sender === 'ai' && lastMessage.isStreaming) {
          lastMessage.text = data.content;
          lastMessage.isStreaming = false;
          lastMessage.isError = true;
          renderChat(currentConvoIdx);
        }
        isWaitingForResponse = false;
        sendBtn.disabled = false;
      }
    }

    // HTTP API fallback
    async function sendMessageHTTP(message) {
      try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query: message,
            session_id: currentSessionId,
            stream: false
          })
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data;
      } catch (error) {
        console.error('HTTP API error:', error);
        throw error;
      }
    }

    // Speech synthesis function
    function speakText(text) {
      return new Promise((resolve) => {
        if ('speechSynthesis' in window) {
          if (currentUtterance) {
            speechSynthesis.cancel();
            currentUtterance = null;
          }
          
          isSpeaking = true;
          currentUtterance = new SpeechSynthesisUtterance(text);
          currentUtterance.lang = 'en-US';
          currentUtterance.rate = 1.0;
          currentUtterance.pitch = 1.0;
          
          currentUtterance.onstart = () => {
            isSpeaking = true;
          };
          
          currentUtterance.onend = () => {
            isSpeaking = false;
            currentUtterance = null;
            resolve();
            
            if (continuousMode && !isListening) {
              setTimeout(() => {
                startListening();
              }, 500);
            }
          };
          
          currentUtterance.onerror = () => {
            isSpeaking = false;
            currentUtterance = null;
            resolve();
          };
          
          speechSynthesis.speak(currentUtterance);
        } else {
          resolve();
        }
      });
    }

    function stopSpeaking() {
      if (currentUtterance && speechSynthesis.speaking) {
        speechSynthesis.cancel();
        currentUtterance = null;
        isSpeaking = false;
      }
    }

    function addTypingIndicator() {
      const typingRow = document.createElement('div');
      typingRow.className = 'message-row typing-indicator';
      typingRow.id = 'typing-indicator';
      typingRow.innerHTML = `
        <div class="message">
          <div class="typing-dots">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>
      `;
      chatArea.appendChild(typingRow);
      chatArea.scrollTop = chatArea.scrollHeight;
    }

    function removeTypingIndicator() {
      const typingIndicator = document.getElementById('typing-indicator');
      if (typingIndicator) {
        typingIndicator.remove();
      }
    }

    function renderChat(convoIdx) {
      chatArea.innerHTML = '';
      const convo = conversations[convoIdx];
      convo.messages.forEach(msg => {
        const row = document.createElement('div');
        row.className = 'message-row' + (msg.sender === 'user' ? ' user' : '');
        
        let messageClass = 'message';
        if (msg.isError) {
          messageClass += ' error-message';
        }
        
        let messageContent = msg.text;
        if (msg.isStreaming) {
          messageContent += '<span class="typing-cursor">|</span>';
        }
        
        row.innerHTML = `<div class="${messageClass}">${messageContent}</div><div class="timestamp">${msg.time}</div>`;
        chatArea.appendChild(row);
      });
      chatArea.scrollTop = chatArea.scrollHeight;
    }

    function renderSidebar() {
      convoList.innerHTML = '';
      conversations.forEach((convo, idx) => {
        const firstUserMsg = convo.messages.find(m => m.sender === 'user');
        const summary = firstUserMsg ? firstUserMsg.text.slice(0, 30) + '...' : 'New conversation';
        const started = new Date(convo.started);
        const time = started.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const li = document.createElement('li');
        li.className = idx === currentConvoIdx ? 'active' : '';
        li.innerHTML = `${summary}<br><span style="font-size:0.85em;color:#90caf9;">${started.toLocaleDateString()} ${time}</span>`;
        li.onclick = () => {
          currentConvoIdx = idx;
          currentSessionId = convo.id;
          renderSidebar();
          renderChat(currentConvoIdx);
          
          // Reconnect WebSocket with new session ID
          if (websocket) {
            websocket.close();
          }
          connectWebSocket();
        };
        convoList.appendChild(li);
      });
    }

    // Form submission handler
    chatForm.addEventListener('submit', async function(e) {
      e.preventDefault();
      const msg = chatInput.value.trim();
      if (!msg || isWaitingForResponse) return;

      stopSpeaking();
      isWaitingForResponse = true;
      sendBtn.disabled = true;

      const userMsg = { sender: 'user', text: msg, time: getCurrentTime() };
      conversations[currentConvoIdx].messages.push(userMsg);
      renderChat(currentConvoIdx);
      chatInput.value = '';
      renderSidebar();

      // Add AI message placeholder for streaming
      const aiMsg = { 
        sender: 'ai', 
        text: '', 
        time: getCurrentTime(),
        isStreaming: true 
      };
      conversations[currentConvoIdx].messages.push(aiMsg);
      renderChat(currentConvoIdx);

      try {
        if (isConnected && websocket.readyState === WebSocket.OPEN) {
          // Send via WebSocket for streaming
          websocket.send(JSON.stringify({
            query: msg,
            stream: true
          }));
        } else {
          // Fallback to HTTP API
          console.log('Using HTTP API fallback');
          const response = await sendMessageHTTP(msg);
          
          aiMsg.text = response.response;
          aiMsg.isStreaming = false;
          aiMsg.sources = response.sources;
          renderChat(currentConvoIdx);
          renderSidebar();
          
          await speakText(response.response);
          isWaitingForResponse = false;
          sendBtn.disabled = false;
        }
      } catch (error) {
        console.error('Error sending message:', error);
        aiMsg.text = 'Sorry, I encountered an error. Please try again.';
        aiMsg.isStreaming = false;
        aiMsg.isError = true;
        renderChat(currentConvoIdx);
        isWaitingForResponse = false;
        sendBtn.disabled = false;
      }
    });

    // Speech recognition initialization
    function initializeSpeechRecognition() {
      if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.continuous = false;
        recognition.maxAlternatives = 1;

        recognition.onstart = function() {
          isListening = true;
          micBtn.classList.add('pulsing');
          micBtn.style.background = 'linear-gradient(135deg,#ff5252 0%,#ff1744 100%)';
          micBtn.title = 'Listening... Click to stop';
        };

        recognition.onend = function() {
          isListening = false;
          micBtn.classList.remove('pulsing');
          micBtn.style.background = 'linear-gradient(135deg,#4fc3f7 0%,#2196f3 100%)';
          micBtn.title = continuousMode ? 'Continuous mode ON - Click to stop' : 'Click to start listening';
        };

        recognition.onerror = function(event) {
          console.error('Speech recognition error:', event.error);
          isListening = false;
          micBtn.classList.remove('pulsing');
          micBtn.style.background = 'linear-gradient(135deg,#4fc3f7 0%,#2196f3 100%)';
          
          if (continuousMode && event.error === 'network') {
            setTimeout(() => {
              if (continuousMode && !isListening) {
                startListening();
              }
            }, 2000);
          }
        };

        recognition.onresult = function(event) {
          const transcript = event.results[0][0].transcript.trim();
          if (transcript) {
            chatInput.value = transcript;
            chatForm.requestSubmit();
          }
        };

        return true;
      }
      return false;
    }

    function startListening() {
      if (recognition && !isListening && !isSpeaking) {
        try {
          recognition.start();
        } catch (error) {
          console.error('Failed to start recognition:', error);
        }
      }
    }

    function stopListening() {
      if (recognition && isListening) {
        recognition.stop();
      }
    }

    // Initialize speech recognition
    if (initializeSpeechRecognition()) {
      micBtn.addEventListener('click', function() {
        stopSpeaking();
        
        if (continuousMode) {
          continuousMode = false;
          stopListening();
          micBtn.title = 'Click to start listening';
          micBtn.style.background = 'linear-gradient(135deg,#4fc3f7 0%,#2196f3 100%)';
        } else {
          continuousMode = true;
          micBtn.title = 'Continuous mode ON - Click to stop';
          startListening();
        }
      });
    } else {
      micBtn.disabled = true;
      micBtn.title = 'Speech recognition not supported in this browser.';
    }

    // New conversation handler
    newConvoBtn.addEventListener('click', function() {
      stopSpeaking();
      stopListening();
      continuousMode = false;
      
      currentSessionId = generateSessionId();
      const newConvo = {
        id: currentSessionId,
        started: new Date().toISOString(),
        messages: [
          { sender: 'ai', text: "Hello! I'm Budger, your AI assistant from Cogent Infotech Corporation. How can I help you today?", time: getCurrentTime() }
        ]
      };
      conversations.push(newConvo);
      currentConvoIdx = conversations.length - 1;
      renderSidebar();
      renderChat(currentConvoIdx);
      
      // Reconnect WebSocket with new session ID
      if (websocket) {
        websocket.close();
      }
      connectWebSocket();
    });

    // Initialize the app
    initializeConversations();
    renderSidebar();
    renderChat(currentConvoIdx);
    connectWebSocket();

    // Clean up on page unload
    window.addEventListener('beforeunload', function() {
      stopSpeaking();
      stopListening();
      if (websocket) {
        websocket.close();
      }
    });
  