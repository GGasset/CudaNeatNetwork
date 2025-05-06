#include <stdlib.h>
#include "SinglyLinkedListNode.h"

#pragma once
template <typename keyT, typename valueT>
class HashTable
{
private:
	SinglyLinkedListNode<keyT>** keys;
	SinglyLinkedListNode<valueT>** values;
	bool* contains_values;

	int bucket_count;
public:

	HashTable(int bucket_count)
	{
		this->bucket_count = bucket_count;
		this->keys = (SinglyLinkedListNode<keyT>**)malloc(sizeof(SinglyLinkedListNode<keyT>*) * bucket_count);
		this->values = (SinglyLinkedListNode<valueT>**)malloc(sizeof(SinglyLinkedListNode<valueT>*) * bucket_count);
		this->contains_values = (bool*)malloc(sizeof(bool) * bucket_count);
		for (size_t i = 0; i < bucket_count && contains_values; i++)
		{
			this->contains_values[i] = false;
		}
	}

	~HashTable()
	{
		this->free();
	}

	SinglyLinkedListNode<keyT>* GetKeys()
	{
		auto output = SinglyLinkedListNode<keyT>(0);
		for (size_t i = 0; i < bucket_count; i++)
		{
			if (contains_values[i])
				output.AddRange(keys[i]->Clone());
		}
		return output.next;
	}

	size_t GetHash(keyT key)
	{
		if (typeid(keyT) == typeid(short) || typeid(keyT) == typeid(int) || typeid(keyT) == typeid(size_t) || typeid(keyT) == typeid(long) || typeid(keyT) == typeid(long long))
		{
			int hash = key;
			if (key) hash = hash * (hash < bucket_count) + (bucket_count % hash) * (hash >= bucket_count);
			hash -= bucket_count * (hash == bucket_count);

			return hash;
		}
		else
		{
			return 0;
		}
	}

	void Add(keyT key, valueT value)
	{
		size_t bucketI = GetHash(key);
		InsertAtBucket(key, value, bucketI);
	}

	void Remove(keyT key)
	{
		size_t bucket_i = GetHash(key);
		bool is_found = contains_values[bucket_i];
		if (!is_found) return;

		int node_i = keys[bucket_i]->GetValueIndex(key);
		is_found = node_i != -1;
		if (!is_found) return;

		keys[bucket_i] = keys[bucket_i]->RemoveNode(node_i);
		values[bucket_i] = values[bucket_i]->RemoveNode(node_i);

		bool error = (keys[bucket_i] == 0 || values[bucket_i] == 0) && (void*)keys[bucket_i] != (void*)values[bucket_i];
		if (error) throw;

		if (keys[bucket_i] == 0)
			contains_values[bucket_i] = false;
	}

	valueT Get(keyT key, bool& is_found)
	{
		int bucket_i = GetHash(key);
		is_found = contains_values[bucket_i];
		if (!is_found)
			return 0;

		int node_i = keys[bucket_i]->GetValueIndex(key);

		is_found = node_i != -1;
		if (!is_found)
			return 0;

		return GetValueAtBucket(bucket_i, node_i);
	}

	

	void free()
	{
		for (size_t i = 0; i < bucket_count && keys && values; i++)
		{
			keys[i]->free();
			values[i]->free();
		}
		std::free(keys);
		std::free(values);
		std::free(contains_values);
	}

private:
	void InsertAtBucket(keyT key, valueT value, int bucket_i)
	{
		if (!this->contains_values[bucket_i])
		{
			this->keys[bucket_i] = new SinglyLinkedListNode<keyT>(key);
			this->values[bucket_i] = new SinglyLinkedListNode<valueT>(value);
			this->contains_values[bucket_i] = true;
			return;
		}

		this->keys[bucket_i]->GetLastNode()->next = new SinglyLinkedListNode<keyT>(key);
		this->values[bucket_i]->GetLastNode()->next = new SinglyLinkedListNode<valueT>(value);
	}

	valueT GetValueAtBucket(int bucket_i, int node_i)
	{
		return values[bucket_i]->GetIndex(node_i)->value;
	}
};
