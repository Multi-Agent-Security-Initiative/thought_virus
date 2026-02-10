"""Data models for multi-agent conversations."""

from dataclasses import dataclass
from typing import Dict, List, Literal


@dataclass
class Message:
    """A single message in a conversation.

    Attributes:
        role: The role of the message sender (system, user, or assistant)
        content: The text content of the message
    """
    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self) -> Dict[str, str]:
        """Convert message to dictionary format for JSON serialization.

        Returns:
            Dictionary with 'role' and 'content' keys
        """
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "Message":
        """Create a Message from a dictionary.

        Args:
            data: Dictionary with 'role' and 'content' keys

        Returns:
            Message instance

        Raises:
            ValueError: If role is not valid or required keys are missing
        """
        if "role" not in data or "content" not in data:
            raise ValueError(f"Message dict must contain 'role' and 'content' keys. Got: {data.keys()}")

        role = data["role"]
        if role not in ["system", "user", "assistant"]:
            raise ValueError(f"Invalid role: {role}. Must be 'system', 'user', or 'assistant'")

        return cls(role=role, content=data["content"])

    def is_system(self) -> bool:
        """Check if this is a system message."""
        return self.role == "system"

    def is_user(self) -> bool:
        """Check if this is a user message."""
        return self.role == "user"

    def is_assistant(self) -> bool:
        """Check if this is an assistant message."""
        return self.role == "assistant"


@dataclass
class Conversation:
    """A conversation history for a single agent.

    Attributes:
        agent_number: The ID of the agent
        messages: List of messages in the conversation
    """
    agent_number: int
    messages: List[Message]

    def to_dict(self) -> Dict[str, any]:
        """Convert conversation to dictionary format for JSON serialization.

        Returns:
            Dictionary with 'agent_number' and 'messages' keys
        """
        return {
            "agent_number": self.agent_number,
            "messages": [msg.to_dict() for msg in self.messages]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "Conversation":
        """Create a Conversation from a dictionary.

        Args:
            data: Dictionary with 'agent_number' and 'messages' keys

        Returns:
            Conversation instance
        """
        agent_number = data.get("agent_number", 0)
        messages = [Message.from_dict(msg) for msg in data.get("messages", [])]
        return cls(agent_number=agent_number, messages=messages)

    @classmethod
    def from_message_list(cls, agent_number: int, messages: List[Dict[str, str]]) -> "Conversation":
        """Create a Conversation from a list of message dictionaries.

        Args:
            agent_number: The ID of the agent
            messages: List of message dictionaries

        Returns:
            Conversation instance
        """
        message_objects = [Message.from_dict(msg) for msg in messages]
        return cls(agent_number=agent_number, messages=message_objects)

    def add_message(self, role: str, content: str) -> None:
        """Add a new message to the conversation.

        Args:
            role: The role of the message sender
            content: The text content of the message
        """
        self.messages.append(Message(role=role, content=content))

    def get_system_prompt(self) -> str:
        """Get the system prompt (first message if it's a system message).

        Returns:
            System prompt content or empty string if no system message
        """
        if self.messages and self.messages[0].is_system():
            return self.messages[0].content
        return ""

    def get_last_message(self) -> Message:
        """Get the last message in the conversation.

        Returns:
            The last message

        Raises:
            IndexError: If conversation is empty
        """
        if not self.messages:
            raise IndexError("Cannot get last message from empty conversation")
        return self.messages[-1]

    def get_messages_as_dicts(self) -> List[Dict[str, str]]:
        """Get all messages as list of dictionaries.

        Returns:
            List of message dictionaries
        """
        return [msg.to_dict() for msg in self.messages]

    def truncate_to_first_exchange(self) -> "Conversation":
        """Create a new conversation with only system, first user, and first assistant messages.

        This is used for "one-way" analysis without the backward pass.

        Returns:
            New Conversation with truncated message history
        """
        if len(self.messages) < 3:
            return Conversation(agent_number=self.agent_number, messages=self.messages.copy())

        # Take first 3 messages (system, user, assistant)
        return Conversation(agent_number=self.agent_number, messages=self.messages[:3])

    def __len__(self) -> int:
        """Get the number of messages in the conversation."""
        return len(self.messages)


@dataclass
class ConversationSet:
    """A set of conversations for all agents in an experiment run.

    Attributes:
        seed: The random seed used for this conversation set
        conversations: Dictionary mapping agent_number to Conversation
    """
    seed: int
    conversations: Dict[int, Conversation]

    def to_dict(self) -> Dict[int, List[Dict[str, str]]]:
        """Convert conversation set to dictionary format for JSON serialization.

        Returns:
            Dictionary mapping agent numbers to lists of message dicts
        """
        return {
            agent_num: conv.get_messages_as_dicts()
            for agent_num, conv in self.conversations.items()
        }

    @classmethod
    def from_dict(cls, seed: int, data: Dict[str, List[Dict[str, str]]]) -> "ConversationSet":
        """Create a ConversationSet from a dictionary.

        Args:
            seed: The random seed for this conversation set
            data: Dictionary mapping agent numbers (as strings) to lists of message dicts

        Returns:
            ConversationSet instance
        """
        conversations = {}
        for agent_num_str, messages in data.items():
            agent_num = int(agent_num_str)
            conversations[agent_num] = Conversation.from_message_list(agent_num, messages)
        return cls(seed=seed, conversations=conversations)

    def get_conversation(self, agent_number: int) -> Conversation:
        """Get conversation for a specific agent.

        Args:
            agent_number: The agent ID

        Returns:
            Conversation for the agent

        Raises:
            KeyError: If agent_number not found
        """
        if agent_number not in self.conversations:
            raise KeyError(f"Agent {agent_number} not found in conversation set")
        return self.conversations[agent_number]

    def __len__(self) -> int:
        """Get the number of agents in this conversation set."""
        return len(self.conversations)
