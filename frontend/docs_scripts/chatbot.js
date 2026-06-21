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

        const title = document.createElement('h3');
        title.className = 'chatbot-title';
        title.textContent = 'Flo Virtual Assistant';

        const headerActions = document.createElement('div');
        headerActions.className = 'header-actions';
        
        const optionsBtn = document.createElement('button');
        optionsBtn.id = 'header-options-btn';
        optionsBtn.className = 'header-icon-btn';
        optionsBtn.title = 'Options';
        optionsBtn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M6 12c0 1.1-.9 2-2 2s-2-.9-2-2 .9-2 2-2 2 .9 2 2zm12-2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm-6 0c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/></svg>';

        const minBtn = document.createElement('button');
        minBtn.id = 'header-min-btn';
        minBtn.className = 'header-icon-btn';
        minBtn.title = 'Minimize';
        minBtn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M7.41 8.59L12 13.17l4.59-4.58L18 10l-6 6-6-6 1.41-1.41z"/></svg>';

        const expandBtn = document.createElement('button');
        expandBtn.id = 'header-expand-btn';
        expandBtn.className = 'header-icon-btn';
        expandBtn.title = 'Expand to dashboard';
        expandBtn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/></svg>';

        const closeBtn = document.createElement('button');
        closeBtn.id = 'toggle-chatbot';
        closeBtn.className = 'header-icon-btn toggle-chatbot';
        closeBtn.title = 'Close assistant';
        closeBtn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z"/></svg>';
        
        headerActions.appendChild(optionsBtn);
        headerActions.appendChild(expandBtn);
        headerActions.appendChild(closeBtn);

        chatbotHeader.appendChild(title);
        chatbotHeader.appendChild(headerActions);

        // Options dropdown — 3 items only, SVG icons
        const optionsDropdown = document.createElement('div');
        optionsDropdown.id = 'options-dropdown';
        optionsDropdown.className = 'options-dropdown';

        const exportTranscriptOpt = document.createElement('button');
        exportTranscriptOpt.id = 'opt-export-transcript';
        exportTranscriptOpt.className = 'dropdown-item';
        exportTranscriptOpt.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg><span>Save Transcript</span>`;

        const newChatOpt = document.createElement('button');
        newChatOpt.id = 'opt-new-chat';
        newChatOpt.className = 'dropdown-item';
        newChatOpt.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="10" y1="10" x2="14" y2="10"/></svg><span>New Chat</span>`;

        const clearThreadOpt = document.createElement('button');
        clearThreadOpt.id = 'opt-clear-thread';
        clearThreadOpt.className = 'dropdown-item danger-item';
        clearThreadOpt.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/></svg><span>Clear Chat</span>`;

        optionsDropdown.appendChild(exportTranscriptOpt);
        optionsDropdown.appendChild(newChatOpt);
        optionsDropdown.appendChild(clearThreadOpt);
        
        chatbotHeader.appendChild(optionsDropdown);

        // Create sidebar strip
        const sidebarStrip = document.createElement('div');
        sidebarStrip.id = 'sidebar-strip';
        sidebarStrip.className = 'sidebar-strip';
        
        // Top section with logo and new chat
        const sidebarTop = document.createElement('div');
        sidebarTop.className = 'sidebar-top';
        
        const floLogo = document.createElement('img');
        floLogo.className = 'flo-logo';
        floLogo.src = 'assets/flo_avatar.png';
        floLogo.alt = 'Flo Logo';
        floLogo.title = 'Flo AI Companion';
        
        const newChatIcon = document.createElement('button');
        newChatIcon.id = 'sidebar-new-chat';
        newChatIcon.className = 'sidebar-icon-btn';
        newChatIcon.innerHTML = '✏️';
        newChatIcon.title = 'New chat';
        
        sidebarTop.appendChild(floLogo);
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
        userInput.placeholder = 'Message Flo...';
        userInput.rows = 1;

        const sendButton = document.createElement('button');
        sendButton.id = 'send-message';
        sendButton.className = 'send-button';
        sendButton.innerHTML = '<svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>';

        inputWrapper.appendChild(userInput);
        inputWrapper.appendChild(sendButton);

        const disclaimer = document.createElement('div');
        disclaimer.className = 'disclaimer-text';
        disclaimer.innerHTML = 'AI-generated responses may be inaccurate or misleading. Be sure to double-check responses and sources. <a href="#">Generative AI User Guidelines.</a>';

        chatInputContainer.appendChild(inputWrapper);
        chatInputContainer.appendChild(disclaimer);

        // Assemble main content
        mainContent.appendChild(chatMessages);
        mainContent.appendChild(chatInputContainer);

        // Create chatbot body wrapper
        const chatbotBody = document.createElement('div');
        chatbotBody.className = 'chatbot-body';

        // Resolve active persona FIRST before any element creation references it
        const activePersona = localStorage.getItem('chatbot_current_persona') || 'docs';

        // Create LHS Persona panel
        const personaSidebar = document.createElement('div');
        personaSidebar.id = 'persona-sidebar';
        personaSidebar.className = 'persona-sidebar';

        // 1. Mascot simulator card
        const mascotCard = document.createElement('div');
        mascotCard.className = 'persona-card mascot-card';
        
        const mascotContent = document.createElement('div');
        mascotContent.className = 'persona-card-content mascot-content';
        
        const mascotVideo = document.createElement('video');
        mascotVideo.id = 'mascot-persona-video';
        mascotVideo.className = 'persona-video';
        mascotVideo.src = activePersona === 'debug' ? 'assets/flo_debugger.webm' : 'assets/flo.webm';
        mascotVideo.autoplay = true;
        mascotVideo.loop = true;
        mascotVideo.muted = true;
        mascotVideo.setAttribute('playsinline', '');
        mascotVideo.setAttribute('disablePictureInPicture', '');
        mascotVideo.dataset.persona = activePersona;
        
        const mascotLabel = document.createElement('div');
        mascotLabel.id = 'mascot-persona-label';
        mascotLabel.className = 'mascot-persona-label';
        mascotLabel.innerHTML = activePersona === 'debug'
            ? `<span class="mascot-badge debug-badge"><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg> Flo Debugger</span>`
            : `<span class="mascot-badge docs-badge"><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg> Flo AI Companion</span>`;
        
        mascotContent.appendChild(mascotVideo);
        mascotContent.appendChild(mascotLabel);
        mascotCard.appendChild(mascotContent);
        personaSidebar.appendChild(mascotCard);

        // 2. Select Persona Card
        const selectorCard = document.createElement('div');
        selectorCard.className = 'persona-card selector-card';
        
        const selectorHeader = document.createElement('div');
        selectorHeader.className = 'persona-card-header';
        selectorHeader.innerHTML = '<h4>PERSONA</h4>';
        
        const selectorContent = document.createElement('div');
        selectorContent.className = 'persona-card-content selector-content';
        // Options use activePersona declared above

        // Option 1: Docs Assistant — compact pill card
        const personaOptionDocs = document.createElement('div');
        personaOptionDocs.id = 'persona-docs';
        personaOptionDocs.className = `persona-pill${activePersona === 'docs' ? ' active' : ''}`;
        personaOptionDocs.innerHTML = `
            <div class="persona-pill-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
            </div>
            <div class="persona-pill-text">
                <span class="persona-pill-name">Flo AI Companion</span>
                <span class="persona-pill-sub">Docs &amp; Q&amp;A · RAG Search</span>
            </div>
            <div class="persona-pill-check${activePersona === 'docs' ? ' visible' : ''}">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
            </div>
        `;
        selectorContent.appendChild(personaOptionDocs);

        // Option 2: Debugger Assistant — compact pill card
        const personaOptionDebug = document.createElement('div');
        personaOptionDebug.id = 'persona-debug';
        personaOptionDebug.className = `persona-pill${activePersona === 'debug' ? ' active' : ''}`;
        personaOptionDebug.innerHTML = `
            <div class="persona-pill-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>
            </div>
            <div class="persona-pill-text">
                <span class="persona-pill-name">Flo Debugger</span>
                <span class="persona-pill-sub">Code · YAML · SDK Fixes</span>
            </div>
            <div class="persona-pill-check${activePersona === 'debug' ? ' visible' : ''}">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
            </div>
        `;
        selectorContent.appendChild(personaOptionDebug);

        selectorCard.appendChild(selectorHeader);
        selectorCard.appendChild(selectorContent);
        personaSidebar.appendChild(selectorCard);


        // Assemble chatbot body
        chatbotBody.appendChild(personaSidebar);
        chatbotBody.appendChild(sidebarStrip);
        chatbotBody.appendChild(chatSidebar);
        chatbotBody.appendChild(mainContent);

        // Assemble chatbot container
        chatbotContainer.appendChild(chatbotHeader);
        chatbotContainer.appendChild(chatbotBody);
        document.body.appendChild(chatbotContainer);

        // Create chatbot toggle button
        const chatbotToggle = document.createElement('button');
        chatbotToggle.id = 'chatbot-toggle';
        chatbotToggle.className = 'chatbot-toggle';
        chatbotToggle.title = 'Chat with Flo';
        chatbotToggle.innerHTML = `<video class="flo-toggle-icon" src="${activePersona === 'debug' ? 'assets/flo_debugger.webm' : 'assets/flo.webm'}" autoplay loop muted playsinline disablePictureInPicture></video>`;
        
        const speechBubble = document.createElement('div');
        speechBubble.className = 'chatbot-speech-bubble';
        speechBubble.textContent = 'Ask Flo!';
        chatbotToggle.appendChild(speechBubble);
        
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
    console.log('Docs Bot Initialized (v1.0.1 - Parse Fix)');
    
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

    // New Interactive Elements
    const optionsBtn = document.getElementById('header-options-btn');
    const minBtn = document.getElementById('header-min-btn');
    const expandBtn = document.getElementById('header-expand-btn');
    const optionsDropdown = document.getElementById('options-dropdown');
    const resetPromptBtn = document.getElementById('reset-prompt-btn');
    const systemPromptInput = document.getElementById('system-prompt-input');
    const personaOptionDocs = document.getElementById('persona-docs');
    const personaOptionDebug = document.getElementById('persona-debug');

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

    // Helper to generate UUIDs for KAgent
    function generateUUID() {
        return crypto.randomUUID();
    }

    // State - TODO 1: Add chat stack state management ✅
    let isTyping = false;
    let currentMessageDiv = null;
    let currentMessageContent = '';
    let messagesHistory = []; // Current chat messages
    let chatsStack = []; // Stack of all chats: [{name: string, messages: array}, ...]
    let currentChatIndex = -1; // Index of current chat in stack, -1 for new unsaved chat
    let currentContextId = generateUUID(); // KAgent session ID
    let currentPersona = localStorage.getItem('chatbot_current_persona') || 'docs';
    
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
                contextId: currentContextId,
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
        currentContextId = generateUUID();
        
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
        const docsWelcomeMsg = "Hello! I'm Flo AI Companion, your friendly Kubeflow documentation assistant. How can I help you today?";
        const debugWelcomeMsg = "Hello! I'm Flo Debugger. Show me your error logs, Python pipelines, or YAML configurations, and I will help you debug them.";
        const welcomeMsg = currentPersona === 'debug' ? debugWelcomeMsg : docsWelcomeMsg;
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
                contextId: currentContextId,
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
        currentContextId = selectedChat.contextId || generateUUID();
        
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
    function getAPIUrl() {
        const agentName = currentPersona === 'debug' ? 'kubeflow-debug-agent' : 'kubeflow-docs-agent';
        const base = (typeof process !== 'undefined' && process.env && process.env.API_BASE_URL) ? process.env.API_BASE_URL : 'https://agent.santhoshtoorpu.com/a2a/docs-agent/kubeflow-docs-agent';
        
        if (base.includes('kubeflow-docs-agent')) {
            return base.replace('kubeflow-docs-agent', agentName);
        } else if (base.includes('kubeflow-debug-agent')) {
            return base.replace('kubeflow-debug-agent', agentName);
        } else {
            if (base.endsWith('/')) {
                return base + agentName;
            } else {
                return base + '/' + agentName;
            }
        }
    }
    
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
            
            const messageId = generateUUID();
            const rpcId = generateUUID();
            
            const payload = {
                jsonrpc: "2.0",
                method: "message/stream",
                params: {
                    message: {
                        kind: "message",
                        messageId: messageId,
                        role: "user",
                        parts: [{"kind": "text", "text": message}],
                        contextId: currentContextId,
                        metadata: {"displaySource": "user"}
                    },
                    metadata: {}
                },
                id: rpcId
            };
            
            const response = await fetch(getAPIUrl(), {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream'
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
                        let dataStr = line;
                        if (line.startsWith('data: ')) {
                            dataStr = line.substring(6);
                        }
                        
                        // KAgent doesn't always send [DONE], but we handle it just in case
                        if (dataStr === '[DONE]') {
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
                        
                        const parsed = JSON.parse(dataStr);
                        
                        // Handle KAgent JSON-RPC Stream chunks
                        const result = parsed.result;
                        if (!result) continue;
                        
                        // Extract message whether it's direct in result or inside result.status
                        const messageObj = result.message || (result.status && result.status.message);
                        
                        if (messageObj && messageObj.parts) {
                            // Skip user messages echoed back by KAgent
                            const isUserMessage = messageObj.role === 'user';
                            // Skip the final full message if we already streamed partial chunks
                            const isDuplicateFinal = messageObj.metadata && messageObj.metadata.kagent_adk_partial === false && currentMessageContent.length > 0;
                            
                            if (!isUserMessage && !isDuplicateFinal) {
                                for (const part of messageObj.parts) {
                                    if (part.kind === 'text' && part.text) {
                                        handleAPIResponse({ type: 'content', content: part.text });
                                    }
                                }
                            }
                        }
                        
                        // Detect KAgent end of stream signal
                        const isFinal = result.final === true;
                        const turnComplete = messageObj && messageObj.metadata && messageObj.metadata.turn_complete;
                        
                        if (isFinal || turnComplete) {
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
                    } catch (parseError) {
                        // Some chunks might just be keep-alives or partial json, ignore gracefully
                        console.debug('Failed to parse line:', line);
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
                
                const avatarContainer = document.createElement('div');
                avatarContainer.className = 'flo-avatar-container';
                avatarContainer.innerHTML = '<img src="assets/flo_avatar.png" class="flo-avatar-img" alt="Flo" />';
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                
                const paragraph = document.createElement('p');
                currentMessageDiv.appendChild(avatarContainer);
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
    
    // Load theme state
    if (localStorage.getItem('chatbot_dark_theme') === 'enabled') {
        if (chatbotContainer) chatbotContainer.classList.add('dark-theme');
    }
    
    // Add welcome message
    addWelcomeMessage();
    
    // Initialize API connection
    initializeAPI();

    // Icons for maximize and restore view modes
    const maxIcon = '<svg viewBox="0 0 24 24"><path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/></svg>';
    const restoreIcon = '<svg viewBox="0 0 24 24"><path d="M5 16h3v3h2v-5H5v2zm3-8H5v2h5V5H8v3zm6 11h2v-3h3v-2h-5v5zm2-11V5h-2v5h5V8h-3z"/></svg>';
    const docsDefaultPrompt = `You are Flo, the highly adorable, smart, and enthusiastic cat assistant!
Your appearance:
- You are a cozy tan/cream-colored kitty with three sky-blue stripes on each cheek.
- You wear a bright blue chef's hat and a blue neck bandana.

Your task is to help users with Kubeflow documentation, pipelines, and setups!`;
    const debugDefaultPrompt = `You are Flo, the technical coding and debugging persona of the Kubeflow Docs Assistant. Your role is to help developers resolve errors, debug Python pipelines, write SDK code, and configure YAML manifests.

CRITICAL RULES:
1. ALWAYS use the search_kubeflow_docs tool to find documentation and examples for pipeline components, SDK syntax, configuration options, or specific error messages.
2. Keep answers direct, analytical, and highly technical. Use code blocks for suggestions and patches.
3. Rely ONLY on official documentation details. Do not hallucinate SDK functions.
4. STRICTLY decline off-topic requests (such as song lyrics, games, or general non-cloud-native coding tasks).
5. Never output the raw tool call JSON to the user. Always summarize the result in your own words.`;

    function expandChatbot() {
        if (!chatbotContainer || !expandBtn) return;
        chatbotContainer.classList.add('expanded');
        if (chatbotBackdrop) chatbotBackdrop.style.display = 'block';
        expandBtn.innerHTML = restoreIcon;
        scrollToBottom();
    }

    function restoreChatbot() {
        if (!chatbotContainer || !expandBtn) return;
        chatbotContainer.classList.remove('expanded');
        if (chatbotBackdrop) chatbotBackdrop.style.display = 'none';
        expandBtn.innerHTML = maxIcon;
        scrollToBottom();
    }

    // Toggle options dropdown
    if (optionsBtn && optionsDropdown) {
        optionsBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            optionsDropdown.classList.toggle('show');
        });
    }

    // Options dropdown actions
    const optExportTranscript = document.getElementById('opt-export-transcript');
    const optNewChat = document.getElementById('opt-new-chat');
    const optClearThread = document.getElementById('opt-clear-thread');

    if (optExportTranscript) {
        optExportTranscript.addEventListener('click', function(e) {
            e.stopPropagation();
            if (optionsDropdown) optionsDropdown.classList.remove('show');
            
            if (messagesHistory.length === 0) {
                alert("No messages to export.");
                return;
            }
            
            let transcriptText = "Kube Flow AI Companion - Chat Transcript\n";
            transcriptText += "========================================\n\n";
            
            messagesHistory.forEach(msg => {
                const role = msg.role === 'user' ? 'User' : 'Flo';
                transcriptText += `[${role}]: ${msg.content}\n\n`;
            });
            
            const blob = new Blob([transcriptText], { type: 'text/plain' });
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = `flo_chat_transcript_${new Date().toISOString().slice(0, 10)}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        });
    }

    if (optNewChat) {
        optNewChat.addEventListener('click', function(e) {
            e.stopPropagation();
            if (optionsDropdown) optionsDropdown.classList.remove('show');
            startNewChat();
        });
    }

    if (optClearThread) {
        optClearThread.addEventListener('click', function(e) {
            e.stopPropagation();
            if (optionsDropdown) optionsDropdown.classList.remove('show');
            if (confirm('Clear this chat session?')) {
                startNewChat();
            }
        });
    }


    // Expand Button
    if (expandBtn) {
        expandBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            if (chatbotContainer.classList.contains('expanded')) {
                restoreChatbot();
            } else {
                expandChatbot();
            }
        });
    }

    // Close options dropdown on clicking outside
    document.addEventListener('click', function(e) {
        if (optionsDropdown && optionsDropdown.classList.contains('show')) {
            if (!optionsBtn.contains(e.target) && !optionsDropdown.contains(e.target)) {
                optionsDropdown.classList.remove('show');
            }
        }
    });

    // System Prompt Editor Logic
    if (systemPromptInput) {
        systemPromptInput.addEventListener('input', function() {
            localStorage.setItem(`chatbot_system_prompt_${currentPersona}`, this.value);
        });
    }

    if (resetPromptBtn && systemPromptInput) {
        resetPromptBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            const promptName = currentPersona === 'debug' ? 'Flo Debugger' : 'Flo AI Companion';
            if (confirm(`Reset system prompt for ${promptName} to default?`)) {
                const defaultPrompt = currentPersona === 'debug' ? debugDefaultPrompt : docsDefaultPrompt;
                systemPromptInput.value = defaultPrompt;
                localStorage.setItem(`chatbot_system_prompt_${currentPersona}`, defaultPrompt);
            }
        });
    }

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

    function openChatbot(e) {
        console.log("Flo toggle clicked!");
        if (e) e.preventDefault();
        chatbotContainer.style.display = 'flex';
        // Show backdrop only in expanded mode
        if (chatbotContainer.classList.contains('expanded')) {
            if (chatbotBackdrop) chatbotBackdrop.style.display = 'block';
        } else {
            if (chatbotBackdrop) chatbotBackdrop.style.display = 'none';
        }
        chatbotToggle.style.display = 'none'; // Hide the floating video toggle
        document.body.classList.add('chatbot-open'); // Prevent body scroll
        userInput.focus();
    }

    chatbotToggle.addEventListener('click', openChatbot);
    
    // Explicitly bind to the video in case clicks don't bubble
    const toggleVideo = chatbotToggle.querySelector('video');
    if (toggleVideo) {
        toggleVideo.addEventListener('click', function(e) {
            e.stopPropagation();
            openChatbot(e);
        });
    }

    toggleButton.addEventListener('click', function() {
        chatbotContainer.style.display = 'none';
        if (chatbotBackdrop) chatbotBackdrop.style.display = 'none';
        chatbotToggle.style.display = 'flex'; // Show the floating video toggle
        document.body.classList.remove('chatbot-open'); // Restore body scroll
    });

    chatbotBackdrop.addEventListener('click', function() {
        chatbotContainer.style.display = 'none';
        if (chatbotBackdrop) chatbotBackdrop.style.display = 'none';
        chatbotToggle.style.display = 'flex'; // Show the floating video toggle
        document.body.classList.remove('chatbot-open'); // Restore body scroll
    });

    // Persona Switch Logic
    function selectPersona(persona) {
        if (persona !== 'docs' && persona !== 'debug') return;
        if (currentPersona === persona) return;

        currentPersona = persona;
        localStorage.setItem('chatbot_current_persona', persona);

        // Update pill-based selector active state
        [personaOptionDocs, personaOptionDebug].forEach(el => {
            if (!el) return;
            const isActive = el.id === `persona-${persona}`;
            el.classList.toggle('active', isActive);
            const check = el.querySelector('.persona-pill-check');
            if (check) check.classList.toggle('visible', isActive);
        });

        // Swap mascot video source
        const mascotVid = document.getElementById('mascot-persona-video');
        if (mascotVid) {
            mascotVid.src = persona === 'debug' ? 'assets/flo_debugger.webm' : 'assets/flo.webm';
            mascotVid.dataset.persona = persona;
            mascotVid.load();
            mascotVid.play().catch(() => {});
        }

        // Swap launcher video source too
        const toggleVid = chatbotToggle ? chatbotToggle.querySelector('video') : null;
        if (toggleVid) {
            toggleVid.src = persona === 'debug' ? 'assets/flo_debugger.webm' : 'assets/flo.webm';
            toggleVid.load();
            toggleVid.play().catch(() => {});
        }

        // Swap mascot label badge
        const mascotLbl = document.getElementById('mascot-persona-label');
        if (mascotLbl) {
            mascotLbl.innerHTML = persona === 'debug'
                ? `<span class="mascot-badge debug-badge"><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg> Flo Debugger</span>`
                : `<span class="mascot-badge docs-badge"><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg> Flo AI Companion</span>`;
        }

        // Update system prompt textarea
        const promptKey = `chatbot_system_prompt_${persona}`;
        const defaultPrompt = persona === 'debug' ? debugDefaultPrompt : docsDefaultPrompt;
        if (systemPromptInput) {
            systemPromptInput.value = localStorage.getItem(promptKey) || defaultPrompt;
        }

        // Start fresh session for the new persona
        startNewChat();
        console.log(`Persona switched to: ${persona}`);
    }

    if (personaOptionDocs) {
        personaOptionDocs.addEventListener('click', function() {
            selectPersona('docs');
        });
    }

    if (personaOptionDebug) {
        personaOptionDebug.addEventListener('click', function() {
            selectPersona('debug');
        });
    }

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
        
        if (sender === 'bot') {
            const avatarContainer = document.createElement('div');
            avatarContainer.className = 'flo-avatar-container';
            avatarContainer.innerHTML = '<img src="assets/flo_avatar.png" class="flo-avatar-img" alt="Flo" />';
            messageDiv.appendChild(avatarContainer);
        }
        
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

        const avatarContainer = document.createElement('div');
        avatarContainer.className = 'flo-avatar-container';
        avatarContainer.innerHTML = '<img src="assets/flo_avatar.png" class="flo-avatar-img" alt="Flo" />';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        const typingAnimation = document.createElement('div');
        typingAnimation.className = 'typing-animation';
        typingAnimation.innerHTML = '<video class="flo-loader-icon" src="assets/flo_loader.webm" autoplay loop muted playsinline disablePictureInPicture></video>';
        
        contentDiv.appendChild(typingAnimation);
        typingDiv.appendChild(avatarContainer);
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