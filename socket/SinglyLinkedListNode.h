#include <typeinfo>

#pragma once
template <typename T>
class SinglyLinkedListNode
{
public:
	SinglyLinkedListNode(T value)
	{
		this->value = value;
		this->next = 0;
	}

	SinglyLinkedListNode* next;
	T value;

	SinglyLinkedListNode* operator[](int i)
	{
		return GetIndex(i);
	}

	SinglyLinkedListNode* GetIndex(size_t i)
	{
		if (i == 0)
		{
			return this;
		}

		if (!this->next)
		{
			return NULL;
		}

		return next->GetIndex(i - 1);
	}

	T get_max()
	{
		if (!next) return value;
		
		T foward_max = next->get_max();
		return value * (value > foward_max) + foward_max * (foward_max >= value);
	}

	size_t GetLastIndex(T value)
	{
		int output = -1;

		SinglyLinkedListNode<T>* current = this;
		size_t i = 0;
		while (current)
		{
			output += (i - output) * (value == current->value);

			i++;
			current = current->next;
		}

		return output;
	}

	int GetValueIndex(T value, int i = 0)
	{
		if (this->value == value)
			return i;
		if (!this->next)
			return -1;
		return this->next->GetValueIndex(value, i++);
	}

	size_t Count(T value, size_t starting_count = 0)
	{
		starting_count += value == this->value;
		if (this->next)
		{
			starting_count += this->next->Count(value, starting_count);
		}
		return starting_count;
	}

	SinglyLinkedListNode* GetLastNode()
	{
		if (!this->next)
		{
			return this;
		}
		return this->next->GetLastNode();
	}

	SinglyLinkedListNode* GetSecondToLastNode()
	{
		if (!this->next)
		{
			return this;
		}
		if (!this->next->next)
		{
			return this;
		}
		return GetSecondToLastNode();
	}

	SinglyLinkedListNode* Reverse(SinglyLinkedListNode* firstNode = NULL)
	{
		SinglyLinkedListNode* new_first_node = new SinglyLinkedListNode(this->value);

		new_first_node->next = firstNode;

		if (!this->next)
		{
			return new_first_node;
		}

		if (!firstNode)
		{
			firstNode = new_first_node;
			return this->next->Reverse(firstNode);
		}

		return this->next->Reverse(new_first_node);
	}

	SinglyLinkedListNode<T>* Clone()
	{
		auto new_node = new SinglyLinkedListNode<T>(value);
		if (next)
			new_node->next = next->Clone();
		return new_node;
	}
	
	void AddRange(SinglyLinkedListNode<T>* start)
	{
		GetLastNode()->next = start;
	}

	T* ToArray(bool free_nodes = false, size_t i = 0, T* array = 0)
	{
		if (!array)
		{
			size_t list_length = GetLength();
			array = new T[list_length];
		}

		array[i] = this->value;
		if (this->next)
		{
			this->next->ToArray(false, i + 1, array);
		}
		if (free_nodes)
		{
			this->free();
		}
		return array;
	}

	SinglyLinkedListNode* RemoveNode(size_t remove_i)
	{
		SinglyLinkedListNode<T>* current_node = this;
		if (!remove_i)
		{
			SinglyLinkedListNode<T>* next = current_node->next;
			delete this;
			return next;
		}
		size_t i = 0;
		for (i = 0; i < remove_i - 1 && current_node->next; i++, current_node = current_node->next) {}
		if (i != remove_i - 1 && current_node->next)
			return this;

		SinglyLinkedListNode<T>* to_remove = current_node->next;
		current_node->next = to_remove->next;
		delete to_remove;
		return this;
	}

	void free()
	{
		if (this->next) this->next->free();
		std::free(this);
	}

	size_t GetLength(size_t i = 1)
	{
		if (!this->next)
		{
			return i;
		}
		return this->next->GetLength(i + 1);
	}
};
