// Function to create chatbot HTML elements dynamically
function createChatbotElements() {
    try {
        // Check if elements already exist (prevent duplicates)
        if (document.getElementById('chatbot-container')) {
            console.log('Chatbot elements already exist, skipping creation');
            return true;
        }

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

        // Create sidebar strip
        const sidebarStrip = document.createElement('div');
        sidebarStrip.id = 'sidebar-strip';
        sidebarStrip.className = 'sidebar-strip';
        
        // Top section with logo and new chat
        const sidebarTop = document.createElement('div');
        sidebarTop.className = 'sidebar-top';
        
        const kubeflowLogo = document.createElement('img');
        kubeflowLogo.className = 'kubeflow-logo';
        kubeflowLogo.src = 'https://raw.githubusercontent.com/kubeflow/website/master/static/favicon-32x32.png';
        kubeflowLogo.alt = 'Kubeflow Logo';
        kubeflowLogo.title = 'Kubeflow Docs Bot';
        
        const newChatIcon = document.createElement('button');
        newChatIcon.id = 'sidebar-new-chat';
        newChatIcon.className = 'sidebar-icon-btn';
        newChatIcon.innerHTML = '✏️';
        newChatIcon.title = 'New chat';
        
        sidebarTop.appendChild(kubeflowLogo);
        sidebarTop.appendChild(newChatIcon);
        
        // Bottom section with expand icon
        const sidebarBottom = document.createElement('div');
        sidebarBottom.className = 'sidebar-bottom';
        
        const expandIcon = document.createElement('button');
        expandIcon.id = 'sidebar-expand';
        expandIcon.className = 'sidebar-icon-btn expand-btn';
        expandIcon.innerHTML = '☰';
        expandIcon.title = 'Chat history';
        
        sidebarBottom.appendChild(expandIcon);
        
        sidebarStrip.appendChild(sidebarTop);
        sidebarStrip.appendChild(sidebarBottom);

        // Create expanded sidebar
        const chatSidebar = document.createElement('div');
        chatSidebar.id = 'chat-sidebar';
        chatSidebar.className = 'chat-sidebar collapsed';
        
        const sidebarHeader = document.createElement('div');
        sidebarHeader.className = 'sidebar-header';
        sidebarHeader.innerHTML = '<h3>Chat History</h3>';
        
        const chatList = document.createElement('div');
        chatList.id = 'chat-list';
        chatList.className = 'chat-list';
        
        chatSidebar.appendChild(sidebarHeader);
        chatSidebar.appendChild(chatList);

        // Create main content area
        const mainContent = document.createElement('div');
        mainContent.id = 'main-content';
        mainContent.className = 'main-content';

        // Create chat messages area
        const chatMessages = document.createElement('div');
        chatMessages.id = 'chat-messages';
        chatMessages.className = 'chat-messages';


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

        // Assemble main content
        mainContent.appendChild(chatMessages);
        mainContent.appendChild(chatInputContainer);

        // Assemble chatbot container
        chatbotContainer.appendChild(chatbotHeader);
        chatbotContainer.appendChild(sidebarStrip);
        chatbotContainer.appendChild(chatSidebar);
        chatbotContainer.appendChild(mainContent);
        document.body.appendChild(chatbotContainer);

        // Create chatbot toggle button
        const chatbotToggle = document.createElement('button');
        chatbotToggle.id = 'chatbot-toggle';
        chatbotToggle.className = 'chatbot-toggle';
        chatbotToggle.innerHTML = '<span class="chat-icon">💬</span><span class="chat-text">Docs Bot</span>';
        document.body.appendChild(chatbotToggle);

        // Force a small delay to ensure DOM is updated
        return new Promise(resolve => {
            setTimeout(() => {
                // Verify all elements were created successfully
                const requiredElements = [
                    'chatbot-container', 'chatbot-backdrop', 'chat-messages', 
                    'user-input', 'send-message', 'toggle-chatbot', 'chatbot-toggle', 
                    'sidebar-strip', 'sidebar-new-chat', 'sidebar-expand', 'chat-sidebar', 'chat-list'
                ];
                
                const missingElements = requiredElements.filter(id => !document.getElementById(id));
                if (missingElements.length > 0) {
                    console.error('Failed to create chatbot elements:', missingElements);
                    resolve(false);
                } else {
                    console.log('All chatbot elements created successfully');
                    resolve(true);
                }
            }, 10);
        });
        
    } catch (error) {
        console.error('Error creating chatbot elements:', error);
        return false;
    }
}

document.addEventListener('DOMContentLoaded', async function() {
    // Create chatbot HTML structure dynamically and wait for completion
    const elementsCreated = await createChatbotElements();
    
    if (!elementsCreated) {
        console.error('Failed to create chatbot elements, aborting initialization');
        return;
    }
    
    // DOM Elements - with null checks
    const chatbotContainer = document.getElementById('chatbot-container');
    const chatbotBackdrop = document.getElementById('chatbot-backdrop');
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-message');
    const toggleButton = document.getElementById('toggle-chatbot');
    const chatbotToggle = document.getElementById('chatbot-toggle');
    const sidebarStrip = document.getElementById('sidebar-strip');
    const sidebarNewChat = document.getElementById('sidebar-new-chat');
    const sidebarExpand = document.getElementById('sidebar-expand');
    const chatSidebar = document.getElementById('chat-sidebar');
    const chatList = document.getElementById('chat-list');

    // Validate all required elements exist
    if (!chatbotContainer || !chatbotBackdrop || !chatMessages || !userInput || !sendButton || !toggleButton || !chatbotToggle || !sidebarStrip || !sidebarNewChat || !sidebarExpand || !chatSidebar || !chatList) {
        console.error('Some chatbot elements are missing:', {
            chatbotContainer: !!chatbotContainer,
            chatbotBackdrop: !!chatbotBackdrop,
            chatMessages: !!chatMessages,
            userInput: !!userInput,
            sendButton: !!sendButton,
            toggleButton: !!toggleButton,
            chatbotToggle: !!chatbotToggle,
            sidebarStrip: !!sidebarStrip,
            sidebarNewChat: !!sidebarNewChat,
            sidebarExpand: !!sidebarExpand,
            chatSidebar: !!chatSidebar,
            chatList: !!chatList
        });
        return;
    }

    // State - TODO 1: Add chat stack state management ✅
    let isTyping = false;
    let currentMessageDiv = null;
    let currentMessageContent = '';
    let messagesHistory = []; // Current chat messages
    let chatsStack = []; // Stack of all chats: [{name: string, messages: array}, ...]
    let currentChatIndex = -1; // Index of current chat in stack, -1 for new unsaved chat
    
    // TODO 2: Browser storage functions ✅
    function saveChatsToStorage() {
        try {
            localStorage.setItem('chatbot_chats_stack', JSON.stringify(chatsStack));
            console.log(`Saved ${chatsStack.length} chats to storage`);
        } catch (error) {
            console.error('Error saving chats to storage:', error);
        }
    }
    
    function loadChatsFromStorage() {
        try {
            const saved = localStorage.getItem('chatbot_chats_stack');
            if (saved) {
                chatsStack = JSON.parse(saved);
                console.log(`Loaded ${chatsStack.length} chats from storage`);
            }
        } catch (error) {
            console.error('Error loading chats from storage:', error);
            chatsStack = [];
        }
    }
    
    function generateChatName(messages) {
        // Generate name from first 3 words of first user message
        const firstUserMessage = messages.find(msg => msg.role === 'user');
        if (firstUserMessage) {
            const words = firstUserMessage.content.trim().split(/\s+/);
            const first3Words = words.slice(0, 3).join(' ');
            return first3Words || `Chat ${chatsStack.length + 1}`;
        }
        return `Chat ${chatsStack.length + 1}`;
    }
    
    // TODO 3: New chat functionality ✅
    function startNewChat() {
        // Save current chat to stack if it has messages
        if (messagesHistory.length > 1) { // More than just welcome message
            const chatName = generateChatName(messagesHistory);
            const currentChat = {
                name: chatName,
                messages: [...messagesHistory] // Copy array
            };
            
            if (currentChatIndex === -1) {
                // Push new chat to stack
                chatsStack.push(currentChat);
                console.log(`Pushed new chat to stack: "${chatName}"`);
            } else {
                // Update existing chat in stack
                chatsStack[currentChatIndex] = currentChat;
                console.log(`Updated existing chat: "${chatName}"`);
            }
            
            // Save to storage
            saveChatsToStorage();
        }
        
        // Reset current chat state
        currentChatIndex = -1;
        messagesHistory = [];
        
        // Clear UI
        clearChatUI();
        
        // Add welcome message
        addWelcomeMessage();
        
        // Update sidebar
        updateSidebar();
        
        console.log('Started new chat');
    }
    
    function clearChatUI() {
        if (chatMessages) {
            chatMessages.innerHTML = '';
        }
    }
    
    function addWelcomeMessage() {
        const welcomeMsg = "Hello! I'm your documentation assistant. How can I help you today?";
        addMessage(welcomeMsg, 'bot');
        messagesHistory.push({
            role: 'assistant',
            content: welcomeMsg
        });
    }
    
    // Auto-save current chat periodically
    function autoSaveCurrentChat() {
        if (messagesHistory.length > 1) { // More than just welcome message
            const chatName = generateChatName(messagesHistory);
            const currentChat = {
                name: chatName,
                messages: [...messagesHistory]
            };
            
            if (currentChatIndex === -1) {
                // This is a new chat, push to stack
                chatsStack.push(currentChat);
                currentChatIndex = chatsStack.length - 1;
                console.log(`Auto-saved new chat: "${chatName}"`);
            } else {
                // Update existing chat
                chatsStack[currentChatIndex] = currentChat;
                console.log(`Auto-saved existing chat: "${chatName}"`);
            }
            
            saveChatsToStorage();
            updateSidebar(); // Refresh sidebar to show updated chat names
        }
    }
    
    // Sidebar Management Functions
    function updateSidebar() {
        if (!chatList || !chatSidebar) return;
        
        // Only render chats if sidebar is expanded (not collapsed)
        if (chatSidebar.classList.contains('collapsed')) {
            chatList.innerHTML = ''; // Clear when collapsed
            return;
        }
        
        // Clear existing items
        chatList.innerHTML = '';
        
        // Add each chat to the sidebar
        chatsStack.forEach((chat, index) => {
            const chatItem = document.createElement('div');
            chatItem.className = 'chat-item';
            if (index === currentChatIndex) {
                chatItem.classList.add('active');
            }
            
            const chatName = document.createElement('div');
            chatName.className = 'chat-name';
            chatName.textContent = chat.name;
            
            const chatActions = document.createElement('div');
            chatActions.className = 'chat-actions';
            
            const deleteButton = document.createElement('button');
            deleteButton.className = 'delete-chat-btn';
            deleteButton.innerHTML = '×';
            deleteButton.title = 'Delete chat';
            deleteButton.onclick = (e) => {
                e.stopPropagation();
                deleteChat(index);
            };
            
            chatActions.appendChild(deleteButton);
            chatItem.appendChild(chatName);
            chatItem.appendChild(chatActions);
            
            // Click to switch to this chat
            chatItem.onclick = () => switchToChat(index);
            
            chatList.appendChild(chatItem);
        });
        
        // Add empty state if no chats
        if (chatsStack.length === 0) {
            const emptyState = document.createElement('div');
            emptyState.className = 'empty-state';
            emptyState.textContent = 'No saved chats yet';
            chatList.appendChild(emptyState);
        }
    }
    
    function toggleSidebar() {
        if (!chatSidebar) return;
        
        chatSidebar.classList.toggle('collapsed');
        if (!chatSidebar.classList.contains('collapsed')) {
            updateSidebar(); // Refresh the sidebar content when opening
        }
    }
    
    function closeSidebar() {
        if (!chatSidebar) return;
        
        chatSidebar.classList.add('collapsed');
    }
    
    function switchToChat(chatIndex) {
        if (chatIndex < 0 || chatIndex >= chatsStack.length) return;
        
        // Save current chat if it has changes
        autoSaveCurrentChat();
        
        // Load the selected chat
        const selectedChat = chatsStack[chatIndex];
        currentChatIndex = chatIndex;
        messagesHistory = [...selectedChat.messages];
        
        // Clear and rebuild UI
        clearChatUI();
        messagesHistory.forEach(msg => {
            if (msg.role === 'user') {
                addMessage(msg.content, 'user');
            } else if (msg.role === 'assistant') {
                addMessage(msg.content, 'bot');
            }
        });
        
        // Update sidebar to show active chat
        updateSidebar();
        
        console.log(`Switched to chat: ${selectedChat.name}`);
    }
    
    function deleteChat(chatIndex) {
        if (chatIndex < 0 || chatIndex >= chatsStack.length) return;
        
        const chatToDelete = chatsStack[chatIndex];
        
        // Remove from stack
        chatsStack.splice(chatIndex, 1);
        
        // Update current chat index if needed
        if (currentChatIndex === chatIndex) {
            // If we're deleting the current chat, start a new one
            currentChatIndex = -1;
            messagesHistory = [];
            clearChatUI();
            addWelcomeMessage();
        } else if (currentChatIndex > chatIndex) {
            // Adjust current index if a chat before it was deleted
            currentChatIndex--;
        }
        
        // Save to storage and update sidebar
        saveChatsToStorage();
        updateSidebar();
        
        console.log(`Deleted chat: ${chatToDelete.name}`);
    }

    // API Configuration
    const API_BASE_URL = 'https://129.80.218.9.nip.io/api/agent/chat';
    const AUTH_TOKEN = process.env.AUTH_TOKEN;
    
    // API connection status
    let isConnected = false;
    
    // Initialize API connection
    function initializeAPI() {
        console.log('Initializing API connection...');
        isConnected = true; // API is stateless, so we consider it always "connected"
        console.log('API connection ready');
    }
    
    // Send message to API
    async function sendMessageToAPI(message, messagesHistory) {
        try {
            console.log('Sending message to API:', message);
            
            const payload = {
                message: message,
                stream: true,
                messages: messagesHistory
            };
            
            const response = await fetch(API_BASE_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${AUTH_TOKEN}`
                },
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            // Handle streaming response
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            // Reset current message state
            currentMessageDiv = null;
            currentMessageContent = '';
            
            while (true) {
                const { done, value } = await reader.read();
                
                if (done) {
                    console.log('Stream completed');
                    break;
                }
                
                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.trim() === '') continue;
                    
                    try {
                        // Handle Server-Sent Events format
                        if (line.startsWith('data: ')) {
                            const data = line.substring(6);
                            if (data === '[DONE]') {
                                // End of stream
                                if (currentMessageContent.trim()) {
                                    messagesHistory.push({
                                        role: 'assistant',
                                        content: currentMessageContent.trim()
                                    });
                                }
                                currentMessageDiv = null;
                                currentMessageContent = '';
                                autoSaveCurrentChat();
                                removeTypingIndicator();
                                return;
                            }
                            
                            const response = JSON.parse(data);
                            handleAPIResponse(response);
                        } else {
                            // Try to parse as JSON directly
                            const response = JSON.parse(line);
                            handleAPIResponse(response);
                        }
                    } catch (parseError) {
                        console.warn('Failed to parse response line:', line, parseError);
                    }
                }
            }
            
        } catch (error) {
            console.error('Error sending message to API:', error);
            removeTypingIndicator();
            addMessage('Sorry, there was an error processing your request. Please try again.', 'bot');
        }
    }
    
    function handleAPIResponse(response) {
        console.log('Received API response:', response);
        
        // Handle different response types
        if (response.type === 'system') {
            console.log('System message:', response.content);
            return;
        }
        
        if (response.type === 'citations') {
            addCitations(response.citations);
            return;
        }
        
        if (response.type === 'content') {
            if (!currentMessageDiv) {
                if (!chatMessages) {
                    console.error('Cannot display message: chat messages container not found');
                    return;
                }

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
            
            // Append new content
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
            // Store the complete bot response in conversation history
            if (currentMessageContent.trim()) {
                messagesHistory.push({
                    role: 'assistant',
                    content: currentMessageContent.trim()
                });
            }
            currentMessageDiv = null;
            currentMessageContent = '';
            autoSaveCurrentChat();
        } else if (response.type === 'error') {
            removeTypingIndicator();
            addMessage('Error: ' + response.content, 'bot');
            currentMessageDiv = null;
            currentMessageContent = '';
        }
    }

    // Load chats from storage and initialize
    loadChatsFromStorage();
    
    // Add welcome message
    addWelcomeMessage();
    
    // Initialize API connection
    initializeAPI();

    // Event Listeners
    sendButton.addEventListener('click', handleSendMessage);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });

    // Auto-resize textarea
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });

    // Chatbot toggle functionality
    chatbotToggle.addEventListener('click', function() {
        chatbotContainer.style.display = 'flex';
        chatbotBackdrop.style.display = 'block';
        document.body.classList.add('chatbot-open'); // Prevent body scroll
        userInput.focus();
    });

    toggleButton.addEventListener('click', function() {
        chatbotContainer.style.display = 'none';
        chatbotBackdrop.style.display = 'none';
        document.body.classList.remove('chatbot-open'); // Restore body scroll
    });

    chatbotBackdrop.addEventListener('click', function() {
        chatbotContainer.style.display = 'none';
        chatbotBackdrop.style.display = 'none';
        document.body.classList.remove('chatbot-open'); // Restore body scroll
    });

    // New chat button event listener
    sidebarNewChat.addEventListener('click', function() {
        console.log('Sidebar new chat clicked');
        startNewChat();
    });
    
    // Sidebar expand button event listener
    sidebarExpand.addEventListener('click', function() {
        console.log('Sidebar expand clicked');
        toggleSidebar();
    });
    
    // Click outside sidebar to close
    document.addEventListener('click', function(e) {
        const sidebar = document.getElementById('chat-sidebar');
        const sidebarStrip = document.getElementById('sidebar-strip');
        const expandBtn = document.getElementById('sidebar-expand');
        
        if (sidebar && !sidebar.classList.contains('collapsed')) {
            // Check if click is outside sidebar and sidebar strip
            if (!sidebar.contains(e.target) && !sidebarStrip.contains(e.target) && e.target !== expandBtn) {
                closeSidebar();
            }
        }
    });

    // Auto-save on page unload
    window.addEventListener('beforeunload', function() {
        autoSaveCurrentChat();
    });

    // Auto-save every 30 seconds
    setInterval(autoSaveCurrentChat, 30000);

    // Utility function to format text
    function formatMarkdown(text, isStreaming = false) {
        if (!text) return '';
        
        let formatted = text;
        const codeBlockPlaceholders = [];
        let placeholderIndex = 0;
        
        // Handle code blocks first (triple backticks) and replace with placeholders
        if (isStreaming) {
            // For streaming, be more careful with incomplete code blocks
            const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
            formatted = formatted.replace(codeBlockRegex, function(match, lang, code) {
                const language = lang || 'text';
                const placeholder = `__CODE_BLOCK_${placeholderIndex}__`;
                codeBlockPlaceholders[placeholderIndex] = `<pre><code class="language-${language}">${escapeHtml(code.trim())}</code></pre>`;
                placeholderIndex++;
                return placeholder;
            });
            
            // Handle incomplete code blocks at the end
            const incompleteCodeRegex = /```(\w+)?\n([\s\S]*)$/;
            if (incompleteCodeRegex.test(formatted) && !formatted.endsWith('```')) {
                formatted = formatted.replace(incompleteCodeRegex, function(match, lang, code) {
                    const language = lang || 'text';
                    const placeholder = `__CODE_BLOCK_${placeholderIndex}__`;
                    codeBlockPlaceholders[placeholderIndex] = `<pre><code class="language-${language}">${escapeHtml(code)}</code></pre>`;
                    placeholderIndex++;
                    return placeholder;
                });
            }
        } else {
            // For complete text, handle normally
            const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
            formatted = formatted.replace(codeBlockRegex, function(match, lang, code) {
                const language = lang || 'text';
                const placeholder = `__CODE_BLOCK_${placeholderIndex}__`;
                codeBlockPlaceholders[placeholderIndex] = `<pre><code class="language-${language}">${escapeHtml(code.trim())}</code></pre>`;
                placeholderIndex++;
                return placeholder;
            });
        }
        
        // Handle inline code (single backticks) - avoid already processed code blocks
        formatted = formatted.replace(/`([^`\n]+)`/g, '<code>$1</code>');
        
        // Handle line breaks (only outside code blocks)
        formatted = formatted.replace(/\n/g, '<br>');
        
        // Handle bold text
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Restore code blocks from placeholders
        codeBlockPlaceholders.forEach((codeBlock, index) => {
            formatted = formatted.replace(`__CODE_BLOCK_${index}__`, codeBlock);
        });
        
        return formatted;
    }

    function handleSendMessage() {
        const message = userInput.value.trim();
        if (!message || isTyping) return;

        // Add user message to history
        messagesHistory.push({
            role: 'user',
            content: message
        });

        // Add user message to UI
        addMessage(message, 'user');
        userInput.value = '';
        userInput.style.height = 'auto';

        // Show typing indicator
        showTypingIndicator();

        // Send message to API
        if (isConnected) {
            sendMessageToAPI(message, messagesHistory);
        } else {
            console.error('API not connected');
            removeTypingIndicator();
            addMessage('Unable to connect to the server. Please try again later.', 'bot');
        }
        
        // Auto-save after user message
        autoSaveCurrentChat();
    }

    function addMessage(text, sender) {
        if (!chatMessages) {
            console.error('Cannot add message: chat messages container not found');
            return;
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        const paragraph = document.createElement('p');
        
        // Format the text based on sender
        if (sender === 'bot') {
            paragraph.innerHTML = formatMarkdown(text);
            
            // Apply syntax highlighting after DOM insertion
            setTimeout(() => {
                if (window.Prism) {
                    const codeBlocks = paragraph.querySelectorAll('pre code');
                    codeBlocks.forEach(block => {
                        if (!block.classList.contains('prism-highlighted')) {
                            block.classList.add('prism-highlighted');
                            window.Prism.highlightElement(block);
                        }
                    });
                }
            }, 10);
        } else {
            paragraph.textContent = text;
        }
        
        contentDiv.appendChild(paragraph);
        messageDiv.appendChild(contentDiv);
        
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    function addCitations(citations) {
        if (!citations || citations.length === 0) return;
        
        // Find the last bot message content to attach citations to
        const lastBotMessage = chatMessages.querySelector('.bot-message:last-child');
        if (!lastBotMessage) {
            console.error('No bot message found to attach citations to');
            return;
        }
        
        // Get the message content div inside the bot message
        const messageContent = lastBotMessage.querySelector('.message-content');
        if (!messageContent) {
            console.error('No message content found to attach citations to');
            return;
        }
        
        const citationsDiv = document.createElement('div');
        citationsDiv.className = 'citations-container';
        
        // Create header with title and toggle
        const citationsHeader = document.createElement('div');
        citationsHeader.className = 'citations-header';
        
        const citationsTitle = document.createElement('h4');
        citationsTitle.textContent = `Sources (${citations.length}):`;
        citationsTitle.className = 'citations-title';
        
        const citationsToggle = document.createElement('span');
        citationsToggle.className = 'citations-toggle';
        citationsToggle.textContent = '▼';
        
        citationsHeader.appendChild(citationsTitle);
        citationsHeader.appendChild(citationsToggle);
        
        // Create collapsible content
        const citationsContent = document.createElement('div');
        citationsContent.className = 'citations-content';
        
        const citationsList = document.createElement('ul');
        citationsList.className = 'citations-list';
        
        citations.forEach((citation, index) => {
            const listItem = document.createElement('li');
            const link = document.createElement('a');
            link.href = citation;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            
            // Extract readable URL text (remove protocol and truncate if too long)
            let displayText = citation.replace(/^https?:\/\//, '');
            if (displayText.length > 60) {
                displayText = displayText.substring(0, 57) + '...';
            }
            link.textContent = displayText;
            
            listItem.appendChild(link);
            citationsList.appendChild(listItem);
        });
        
        citationsContent.appendChild(citationsList);
        
        // Add click handler for toggle
        citationsHeader.addEventListener('click', function() {
            const isExpanded = citationsContent.classList.contains('expanded');
            if (isExpanded) {
                citationsContent.classList.remove('expanded');
                citationsToggle.classList.remove('expanded');
            } else {
                citationsContent.classList.add('expanded');
                citationsToggle.classList.add('expanded');
            }
        });
        
        // Assemble the citations container
        citationsDiv.appendChild(citationsHeader);
        citationsDiv.appendChild(citationsContent);
        
        // Attach citations to the message content
        messageContent.appendChild(citationsDiv);
        scrollToBottom();
    }

    function showTypingIndicator() {
        if (isTyping || !chatMessages) return;
        
        isTyping = true;
        
        const typingDiv = document.createElement('div');
        typingDiv.id = 'typing-indicator';
        typingDiv.className = 'message bot-message typing-indicator';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        const typingAnimation = document.createElement('div');
        typingAnimation.className = 'typing-animation';
        typingAnimation.innerHTML = '<span></span><span></span><span></span>';
        
        contentDiv.appendChild(typingAnimation);
        typingDiv.appendChild(contentDiv);
        chatMessages.appendChild(typingDiv);
        
        scrollToBottom();
    }

    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
        isTyping = false;
    }

    function scrollToBottom() {
        if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    console.log('Chatbot initialized with chat stack system');
});