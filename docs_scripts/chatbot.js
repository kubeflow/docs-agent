// Function to create chatbot HTML elements dynamically
function createChatbotElements() {
    // Create chatbot backdrop
    const chatbotBackdrop = document.createElement('div');
    chatbotBackdrop.id = 'chatbot-backdrop';
    chatbotBackdrop.className = 'chatbot-backdrop';
    document.body.appendChild(chatbotBackdrop);

    // Create chatbot container
    const chatbotContainer = document.createElement('div');
    chatbotContainer.id = 'chatbot-container';
    chatbotContainer.className = 'chatbot-container';

    // Create chatbot header
    const chatbotHeader = document.createElement('div');
    chatbotHeader.className = 'chatbot-header';

    const chatbotTitle = document.createElement('div');
    chatbotTitle.className = 'chatbot-title';
    chatbotTitle.innerHTML = '<span>Docs Bot</span>';

    const toggleButton = document.createElement('button');
    toggleButton.id = 'toggle-chatbot';
    toggleButton.className = 'toggle-chatbot';
    toggleButton.innerHTML = '<span class="minimize-icon">×</span>';

    chatbotHeader.appendChild(chatbotTitle);
    chatbotHeader.appendChild(toggleButton);

    // Create chat messages area
    const chatMessages = document.createElement('div');
    chatMessages.id = 'chat-messages';
    chatMessages.className = 'chat-messages';

    // Create welcome message
    const welcomeMessage = document.createElement('div');
    welcomeMessage.className = 'message bot-message welcome-message';
    const welcomeContent = document.createElement('div');
    welcomeContent.className = 'message-content';
    const welcomeText = document.createElement('p');
    welcomeText.textContent = "Hello! I'm your documentation assistant. How can I help you today?";
    welcomeContent.appendChild(welcomeText);
    welcomeMessage.appendChild(welcomeContent);
    chatMessages.appendChild(welcomeMessage);

    // Create input container
    const chatInputContainer = document.createElement('div');
    chatInputContainer.className = 'chat-input-container';

    const inputWrapper = document.createElement('div');
    inputWrapper.className = 'input-wrapper';

    const userInput = document.createElement('textarea');
    userInput.id = 'user-input';
    userInput.className = 'chat-input';
    userInput.placeholder = 'Message Docs Bot...';
    userInput.rows = 1;

    const sendButton = document.createElement('button');
    sendButton.id = 'send-message';
    sendButton.className = 'send-button';
    sendButton.innerHTML = '<span class="send-icon">➤</span>';

    inputWrapper.appendChild(userInput);
    inputWrapper.appendChild(sendButton);

    const inputFooter = document.createElement('div');
    inputFooter.className = 'input-footer';
    const inputHint = document.createElement('span');
    inputHint.className = 'input-hint';
    inputHint.textContent = 'Press Enter to send, Shift + Enter for new line';
    inputFooter.appendChild(inputHint);

    chatInputContainer.appendChild(inputWrapper);
    chatInputContainer.appendChild(inputFooter);

    // Assemble chatbot container
    chatbotContainer.appendChild(chatbotHeader);
    chatbotContainer.appendChild(chatMessages);
    chatbotContainer.appendChild(chatInputContainer);
    document.body.appendChild(chatbotContainer);

    // Create chatbot toggle button
    const chatbotToggle = document.createElement('button');
    chatbotToggle.id = 'chatbot-toggle';
    chatbotToggle.className = 'chatbot-toggle';
    chatbotToggle.innerHTML = '<span class="chat-icon">💬</span><span class="chat-text">Docs Bot</span>';
    document.body.appendChild(chatbotToggle);
}

document.addEventListener('DOMContentLoaded', function() {
    // Create chatbot HTML structure dynamically
    createChatbotElements();
    
    // DOM Elements
    const chatbotContainer = document.getElementById('chatbot-container');
    const chatbotBackdrop = document.getElementById('chatbot-backdrop');
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-message');
    const toggleButton = document.getElementById('toggle-chatbot');
    const chatbotToggle = document.getElementById('chatbot-toggle');

    // State
    let isTyping = false;
    let socket = null;
    let currentMessageDiv = null;
    let currentMessageContent = '';

    // Initialize WebSocket connection
    function connectWebSocket() {
        try {
            console.log('Attempting to connect to WebSocket...');
            socket = new WebSocket('wss://websocket-server-production-9b44.up.railway.app');
            
            socket.onopen = function(e) {
                console.log('WebSocket connection established successfully');
            };
            
            socket.onmessage = function(event) {
                console.log('Received message from server:', event.data);
                try {
                    const response = JSON.parse(event.data);
                    
                    if (response.type === 'system') {
                        addSystemMessage(response.content);
                        return;
                    }
                    
                    if (response.type === 'citations') {
                        addCitations(response.citations);
                        return;
                    }
                    
                    if (response.type === 'content') {
                        if (!currentMessageDiv) {
                            // Create new message div for the first token
                            currentMessageDiv = document.createElement('div');
                            currentMessageDiv.className = 'message bot-message';
                            
                            const contentDiv = document.createElement('div');
                            contentDiv.className = 'message-content';
                            
                            const paragraph = document.createElement('p');
                            currentMessageDiv.appendChild(contentDiv);
                            contentDiv.appendChild(paragraph);
                            
                            chatMessages.appendChild(currentMessageDiv);
                            removeTypingIndicator();
                        }
                        
                        // Append new content to the end (tokens come in reverse order)
                        currentMessageContent += response.content;
                        const paragraph = currentMessageDiv.querySelector('p');
                        
                        // Format streaming content
                        const formattedText = formatMarkdown(currentMessageContent, true);
                        
                        paragraph.innerHTML = formattedText;
                        
                        // Apply syntax highlighting to any new code blocks
                        if (window.Prism) {
                            const codeBlocks = currentMessageDiv.querySelectorAll('pre code');
                            codeBlocks.forEach(block => {
                                if (!block.classList.contains('prism-highlighted')) {
                                    block.classList.add('prism-highlighted');
                                    window.Prism.highlightElement(block);
                                }
                            });
                        }
                        
                        scrollToBottom();
                    }
                    
                    // Handle end of message or errors
                    if (response.type === 'end') {
                        currentMessageDiv = null;
                        currentMessageContent = '';
                    } else if (response.type === 'error') {
                        removeTypingIndicator();
                        addSystemMessage('Error: ' + response.content);
                        currentMessageDiv = null;
                        currentMessageContent = '';
                    }
                    
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                    removeTypingIndicator();
                    addMessage('Sorry, there was an error processing your request.', 'bot');
                }
            };
            
            socket.onclose = function(event) {
                console.log('WebSocket connection closed, code:', event.code, 'reason:', event.reason);
                addSystemMessage('Disconnected from server');
                setTimeout(connectWebSocket, 3000);
            };
            
            socket.onerror = function(error) {
                console.error('WebSocket error:', error);
                addSystemMessage('Connection error');
            };
        } catch (error) {
            console.error('Error creating WebSocket connection:', error);
            addSystemMessage('Failed to connect');
        }
    }

    // Connect to WebSocket when page loads
    connectWebSocket();

    // Event Listeners
    sendButton.addEventListener('click', handleSendMessage);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });

    toggleButton.addEventListener('click', hideChatbot);
    chatbotToggle.addEventListener('click', showChatbot);
    chatbotBackdrop.addEventListener('click', hideChatbot);

    // Auto-resize textarea
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 150) + 'px';
    });

    // Unified markdown formatting function
    function formatMarkdown(text, isStreaming = false) {
        if (!text) return '';
        
        let formatted = text;
        const tempCodeBlocks = [];
        const tempInlineCode = [];
        
        // Handle code blocks first (triple backticks)
        formatted = formatted.replace(/```(\w+)?\n?([\s\S]*?)```/g, (match, lang, code) => {
            const language = lang || 'text';
            const escapedCode = code.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            const block = `<pre><code class="language-${language}">${escapedCode.trim()}</code></pre>`;
            tempCodeBlocks.push(block);
            return `__CODEBLOCK_${tempCodeBlocks.length - 1}__`;
        });
        
        // Handle inline code (single backticks)
        formatted = formatted.replace(/`([^`]+)`/g, (match, code) => {
            const escapedCode = code.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            const inlineCode = `<code>${escapedCode}</code>`;
            tempInlineCode.push(inlineCode);
            return `__INLINECODE_${tempInlineCode.length - 1}__`;
        });
        
        // Escape remaining HTML
        formatted = formatted.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        
        // Restore code blocks and inline code
        tempCodeBlocks.forEach((block, index) => {
            formatted = formatted.replace(`__CODEBLOCK_${index}__`, block);
        });
        
        tempInlineCode.forEach((code, index) => {
            formatted = formatted.replace(`__INLINECODE_${index}__`, code);
        });
        
        // Bold text
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        return formatted;
    }

    function handleSendMessage() {
        const message = userInput.value.trim();
        if (!message || isTyping) return;

        // Reset the current message state
        currentMessageDiv = null;
        currentMessageContent = '';

        // Add user message
        addMessage(message, 'user');
        userInput.value = '';
        userInput.style.height = 'auto';

        // Show typing indicator
        showTypingIndicator();

        // Send message to WebSocket server
        if (socket && socket.readyState === WebSocket.OPEN) {
            try {
                // Send the raw message - the server expects plain text
                console.log('Sending message to server:', message);
                socket.send(message);
            } catch (error) {
                console.error('Error sending message:', error);
                removeTypingIndicator();
                addSystemMessage('Failed to send message');
                addMessage('Error sending your message. Please try again.', 'bot');
            }
        } else {
            // If WebSocket is not connected, show an error message
            console.error('WebSocket not connected. Current state:', socket ? socket.readyState : 'No socket');
            removeTypingIndicator();
            addSystemMessage('Not connected to server');
            addMessage('Unable to connect to the server. Please try again later.', 'bot');
            // Try to reconnect
            connectWebSocket();
        }
    }

    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        const paragraph = document.createElement('p');
        
        // Use enhanced parsing for bot messages to handle markdown
        if (sender === 'bot') {
            const formattedText = formatMarkdown(text);
            paragraph.innerHTML = formattedText;
            
            // Apply syntax highlighting
            if (window.Prism) {
                const codeBlocks = messageDiv.querySelectorAll('pre code');
                codeBlocks.forEach(block => {
                    window.Prism.highlightElement(block);
                });
            }
        } else {
            paragraph.textContent = text;
        }
        
        contentDiv.appendChild(paragraph);
        messageDiv.appendChild(contentDiv);
        
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    function addSystemMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message system-message';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        const paragraph = document.createElement('p');
        paragraph.textContent = text;
        
        contentDiv.appendChild(paragraph);
        messageDiv.appendChild(contentDiv);
        
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    function addCitations(citations) {
        if (!citations || citations.length === 0) return;
        
        // Find the last bot message to attach citations to
        const messages = chatMessages.querySelectorAll('.bot-message:not(.typing-indicator)');
        const lastBotMessage = messages[messages.length - 1];
        
        if (!lastBotMessage) {
            // If no bot message exists, create a standalone citations message
            const citationDiv = document.createElement('div');
            citationDiv.className = 'message bot-message citations-message';
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            
            const citationsContainer = createCitationsDropdown(citations);
            contentDiv.appendChild(citationsContainer);
            citationDiv.appendChild(contentDiv);
            
            chatMessages.appendChild(citationDiv);
        } else {
            // Attach citations to the last bot message
            const contentDiv = lastBotMessage.querySelector('.message-content');
            const existingCitations = contentDiv.querySelector('.citations-container');
            
            // Remove existing citations if any
            if (existingCitations) {
                existingCitations.remove();
            }
            
            const citationsContainer = createCitationsDropdown(citations);
            contentDiv.appendChild(citationsContainer);
        }
        
        scrollToBottom();
    }

    function createCitationsDropdown(citations) {
        const container = document.createElement('div');
        container.className = 'citations-container';
        
        const toggleButton = document.createElement('button');
        toggleButton.className = 'citations-toggle';
        toggleButton.innerHTML = `
            <span class="citations-icon">📚</span>
            <span class="citations-text">Sources (${citations.length})</span>
            <span class="citations-arrow">▼</span>
        `;
        
        const dropdown = document.createElement('div');
        dropdown.className = 'citations-dropdown';
        
        citations.forEach((citation, index) => {
            const link = document.createElement('a');
            link.href = citation;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            link.className = 'citation-link';
            
            // Extract the page title from URL for better display
            const urlParts = citation.split('/');
            const pageName = urlParts[urlParts.length - 1] || urlParts[urlParts.length - 2];
            const displayName = pageName.replace(/-/g, ' ').replace(/_/g, ' ');
            
            link.innerHTML = `
                <span class="citation-number">${index + 1}</span>
                <span class="citation-title">${displayName}</span>
                <span class="citation-url">${citation}</span>
            `;
            
            dropdown.appendChild(link);
        });
        
        // Toggle functionality
        let isOpen = false;
        toggleButton.addEventListener('click', function() {
            isOpen = !isOpen;
            dropdown.classList.toggle('open', isOpen);
            toggleButton.querySelector('.citations-arrow').textContent = isOpen ? '▲' : '▼';
            toggleButton.classList.toggle('open', isOpen);
        });
        
        container.appendChild(toggleButton);
        container.appendChild(dropdown);
        
        return container;
    }

    function showTypingIndicator() {
        isTyping = true;
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot-message typing-indicator';
        typingDiv.id = 'typing-indicator';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        const dots = document.createElement('div');
        dots.className = 'typing-dots';
        dots.innerHTML = '<span></span><span></span><span></span>';
        
        contentDiv.appendChild(dots);
        typingDiv.appendChild(contentDiv);
        
        chatMessages.appendChild(typingDiv);
        scrollToBottom();
    }

    function removeTypingIndicator() {
        isTyping = false;
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showChatbot() {
        chatbotContainer.classList.remove('minimized');
        chatbotContainer.classList.add('active');
        chatbotBackdrop.classList.add('active');
        chatbotToggle.classList.add('hidden');
        toggleButton.querySelector('.minimize-icon').textContent = '×';
        scrollToBottom();
        document.body.style.overflow = 'hidden';
    }

    function hideChatbot() {
        chatbotContainer.classList.remove('active');
        chatbotBackdrop.classList.remove('active');
        chatbotToggle.classList.remove('hidden');
        
        // Restore body scroll
        document.body.style.overflow = '';
        
        setTimeout(() => {
            chatbotContainer.classList.add('minimized');
            toggleButton.querySelector('.minimize-icon').textContent = '+';
        }, 300);
    }

    // Initialize chatbot state
    chatbotContainer.classList.add('minimized');
    toggleButton.querySelector('.minimize-icon').textContent = '+';
});