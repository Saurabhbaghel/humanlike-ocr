from headers import *
# from config import *

device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# training loop
def training(net, num_epochs, batch_size, train_dataloader, val_dataloader, optimizer, metric, loss_fn):
    
    # print the structure of the net
    print(net)

    # for storing the losses for plotting 
    train_loss = []
    val_loss = []
    
    for epoch in range(num_epochs):
        

        last_loss = 0.0
        list_outputs = []
        list_labels = []
        
        net.init_sequence(batch_size)

        # net.train(True)
        print("Epoch {}".format(epoch))
        for i, data in enumerate(train_dataloader):
            running_loss = 0.0
            
            inputs, labels = data[0].to(device_), data[1].to(device_)
            # print(inputs.size())
            # labels = labels.type(torch.float)
            optimizer.zero_grad()

            
            
            training = True
            # print(inputs.s)
            outputs, _ = net(inputs)
            # print(torch.argmax(outputs, dim=1), labels)
            avg_loss = loss_fn(outputs, labels)
            outputs = torch.argmax(outputs, dim=1)

            list_outputs.append(outputs)
            list_labels.append(labels)
            
            # avg_prec = metric(outputs, torch.argmax(labels, dim=1))
            # print(torch.argmax(outputs, dim=1), torch.argmax(labels, dim=1))
            avg_loss.backward(retain_graph = True)

            optimizer.step()

            running_loss += avg_loss.item()
            last_loss = running_loss
            # last_avg_prec = avg_prec
            train_loss.append(avg_loss.item())
        outputs = torch.cat(list_outputs, dim=0)
        labels = torch.cat(list_labels).squeeze()
        print(outputs, labels)

        acc = metric(outputs, labels)
        print("epoch {}, loss {:.3f}, train acc {:.3f}".format(epoch, last_loss, acc))
        torch.save(net, "model_new_printed_{}_{}.pt".format("20-05-23", epoch))
        
        # validation
        list_outputs = []
        list_labels = []
        
        with torch.no_grad():
            running_vloss = 0.0
            for i, data in enumerate(val_dataloader):
                # net.init_sequence(batch_size)

                vinputs, vlabels = data[0].to(device_), data[1].to(device_)
                # vlabels = vlabels.type(torch.float)
                voutputs, _ = net(vinputs)
                list_outputs.append(voutputs)
                list_labels.append(vlabels)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss
                val_loss.append(vloss.item())

            voutputs = torch.cat(list_outputs, dim=0)
            vlabels = torch.cat(list_labels).squeeze()
            vacc = metric(voutputs, vlabels)
            avg_vloss = running_vloss / (i+1)
            print("Loss train {:.3f},  validation {:.3f},  val acc {:.3f}".format(avg_loss, avg_vloss, vacc))

    return train_loss, val_loss