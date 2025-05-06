#include <vector>

#include "unix_sock_interop.h"

int main(int argc, char *argv[])
{
	int socket_fd = connect();
	if (socket_fd == -1)
		return 1;

	NN_manager networks = NN_manager(1000);
	std::vector<int> clients;
	auto pending_messages_sizes = HashTable<int, size_t>(200);
	auto pending_out_messages = HashTable<int, return_specifier*>(200);
	printf("Waiting for connections...\n");
		/*	size_t new_client_fd = accept(socket_fd, 0, 0);
			if (new_client_fd < 0)
			{
				printf("Accept error\n");
				throw;
			}

			clients.push_back(new_client_fd);
			printf("Client connected\n");*/
	while (true)
	{
		fd_set read_fds;
		fd_set write_fds;

		FD_ZERO(&read_fds);
		FD_ZERO(&write_fds);
		
		FD_SET(socket_fd, &read_fds); // Add fd to set

		int max_fd = socket_fd;
		for (auto it = clients.begin(); it != clients.end(); it++)
		{
			int client_fd = *it;
			FD_SET(client_fd, &read_fds);
			max_fd = max_fd * (max_fd >= client_fd) + client_fd * (client_fd > max_fd);
		}
		max_fd++;

		SinglyLinkedListNode<int>* pending_messages_fds = pending_out_messages.GetKeys();
		for (auto it = pending_messages_fds; it; it = it->next)
			if (!FD_ISSET(it->value, &write_fds))
				FD_SET(it->value, &write_fds);
		if (pending_messages_fds) pending_messages_fds->free();

		timeval timeout;
		timeout.tv_sec = 1;
		timeout.tv_usec = 0;
		if (select(max_fd, &read_fds, &write_fds, 0, &timeout) < 0)
		{
			printf("select error\n");
			throw;
		}

		if (FD_ISSET(socket_fd, &read_fds))
		{
			int new_client_fd = accept(socket_fd, 0, 0);
			if (new_client_fd < 0)
			{
				printf("Accept error\n");
				throw;
			}

			clients.push_back(new_client_fd);
			printf("Client connected\n");
		}
		
		for (size_t i = 0; i < clients.size(); i++)
		{
			int fd = clients[i];
			if (FD_ISSET(fd, &read_fds))
			{
				bool has_sent_message_size = false;
				size_t bytes_to_read = pending_messages_sizes.Get(fd, has_sent_message_size);
				bytes_to_read += sizeof(size_t) * (!has_sent_message_size);
				
				char* message = (char *)malloc(bytes_to_read);
				if (!message) throw;
				size_t bytes_read = recv(fd, message, bytes_to_read, 0);
				if (bytes_read < 0 || bytes_read != bytes_to_read) throw;
				if (has_sent_message_size)
				{
					pending_messages_sizes.Remove(fd);
					
					bool has_pending_out_message = false;
					return_specifier* queued_message = pending_out_messages.Get(fd, has_pending_out_message);
					if (has_pending_out_message) delete[] queued_message->return_value;
					pending_out_messages.Remove(fd);

					return_specifier* returned = networks.parse_message(message, bytes_to_read);
					pending_out_messages.Add(fd, returned);
				}
				else
				{
					size_t message_size = *(size_t*)message;
					if (message_size)
					{
						pending_messages_sizes.Add(fd, message_size);
						continue;
					}

					// Handle client disconnect
					__close_sock(fd);
					bool message_exists = false;
					return_specifier* pending_message = pending_out_messages.Get(fd, message_exists);
					if (pending_message)
					{
						if (pending_message->value_count && pending_message->return_value) delete[] pending_message->return_value;
						free(pending_message);
					}

					clients.erase(clients.begin() + i);
					pending_out_messages.Remove(fd);
					pending_messages_sizes.Remove(fd);
					i--;
					printf("Client disconnected.\n");
					continue;
				}
				free(message);
			}
			if (FD_ISSET(fd, &write_fds))
			{
				bool avalible_message = false;
				return_specifier* out_message = pending_out_messages.Get(fd, avalible_message);
				if (!avalible_message) continue;

				char* raw_message = (char *)out_message;
				send(fd, raw_message + sizeof(data_t*), sizeof(return_specifier) - sizeof(data_t*), 0);
				if (out_message->value_count) send(fd, (char *)out_message->return_value, sizeof(data_t) * out_message->value_count, 0);
				

				if (out_message->value_count) delete[] out_message->return_value;
				free(out_message);

				pending_out_messages.Remove(fd);
			}
		}
	}
	unlink(BIND_PATH);
#ifdef _WIN32
	WSACleanup();
#endif
}

int connect()
{
	printf("Setting up socket...\n");

#ifdef _WIN32
	WSADATA details{};
	if (WSAStartup(MAKEWORD(2, 2), &details)) throw;
#endif

	int socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (socket_fd == -1)
	{
		//int err = WSAGetLastError();
		CLEANUP
		throw;
	}
	/*int enabled = 1;
	if (setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, (char *)&enabled, sizeof(int)) == -1)
	{
		__close_sock(socket_fd);
		CLEANUP
		throw;
	}

	FILE* tmp = fopen(BIND_PATH, "w");
	if (!tmp)
	{
		int err = errno;
		__close_sock(socket_fd);
		CLEANUP
		throw;
	}
	else fclose(tmp);*/

	sockaddr_un address{};
	address.sun_family = AF_UNIX;

	const char bind_path[] = BIND_PATH;
	//address.sun_path[0] = 0;
	strncpy(address.sun_path, bind_path, sizeof(address.sun_path));
	
	unlink(BIND_PATH);
	if (bind(socket_fd, (struct sockaddr*)&address, sizeof(address)))
	{
		printf("bind error\n");
		int err = WSAGetLastError();
		__close_sock(socket_fd);
		unlink(BIND_PATH);
		CLEANUP
		throw;
	}
	printf("binding to \"%s\" was succesful\n", BIND_PATH);
	if (listen(socket_fd, 1024)) throw;
	return socket_fd;
}
