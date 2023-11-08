def test( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:
        print("data shape: ", data.size())
        print("target shape: ", target.size())
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, don't bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect ``datagrad``
        data_grad = data.grad.data

        # Restore the data to its original scale
        data_denorm = denorm(data)

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)

        # Reapply normalization
        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)

        # Re-classify the perturbed image
        output = model(perturbed_data_normalized)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

def plot_like_ggplot():
    plt.style.use('ggplot')
    
    
# adv_imgs is a list of arrays
# train_imgs is a tensor of size (n_imgs, 1, 28, 28)
# add the adv_imgs to the train_imgs tensor
def add_adv_imgs_to_train_imgs(adv_imgs, train_imgs):
    adv_imgs = np.array(adv_imgs)
    adv_imgs = np.expand_dims(adv_imgs, axis=1)
    adv_imgs = torch.from_numpy(adv_imgs)
    train_imgs = torch.cat((train_imgs, adv_imgs), 0)
    return train_imgs

# train_dl is a pytorch DataLoader
# set drop_last to True after initializing the train_dl
def set_drop_last_to_true(train_dl):
    train_dl.drop_last = True
    return train_dl

# adv_targets is a list of ints
# train_targets is a tensor of size (n_imgs)
# add the adv_targets to the train_targets tensor
def add_adv_targets_to_train_targets(adv_targets, train_targets):
    adv_targets = np.array(adv_targets)
    adv_targets = torch.from_numpy(adv_targets)
    train_targets = torch.cat((train_targets, adv_targets), 0)
    return train_targets