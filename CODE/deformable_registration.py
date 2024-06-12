import warnings
warnings.filterwarnings("ignore")

import os
import time

import matplotlib.pyplot as plt
import math

import torch
import torch.utils
import torch.optim as optim

import dataloaders as dl
import cost_functions as cf
import utils
import paths

#import prova

from networks import nonrigid_registration_network as nrn

training_path = paths.training_path
validation_path = paths.validation_path
models_path = paths.models_path
figures_path = paths.figures_path
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def training(training_params):
    model_name = training_params['model_name']
    initial_model_name = training_params['initial_model_name']
    batch_size = training_params['batch_size']
    learning_rate = training_params['learning_rate'] 
    num_epochs = training_params['epochs']
    scheduler_rates = training_params['scheduler_rates']
    num_levels = training_params['num_levels']
    inner_iterations_per_level = training_params['inner_iterations_per_level']
    stride = training_params['stride']
    patch_size = training_params['patch_size']
    alphas = training_params['alphas']
    number_of_patches = training_params['number_of_patches']
    cost_function = training_params['cost_function']
    cost_function_params = training_params['cost_function_params']
    print_step = 20
   
    model_save_paths = list()
    models = list()
    parameters = list()
    optimizers = list()
    schedulers = list()

    last_available_level = 0
    for i in range(num_levels):
        model_save_paths.append(os.path.join(models_path, model_name + "_level_" + str(i+1)))
        if initial_model_name is not None:
            try:
                models.append(nrn.load_network(device, path=os.path.join(models_path, initial_model_name + "_level_" + str(i+1))))
                last_available_level = i + 1
            except:
                models.append(nrn.load_network(device, path=os.path.join(models_path, initial_model_name + "_level_" + str(last_available_level))))
        else:
            models.append(nrn.load_network(device))
        parameters.append(models[i].parameters())
        optimizers.append(optim.Adam(parameters[i], learning_rate))
        schedulers.append(optim.lr_scheduler.LambdaLR(optimizers[i], lambda epoch: scheduler_rates[i]**epoch))

    transforms = None
    training_loader = dl.UnsupervisedLoader(training_path, transforms=transforms)
    validation_loader = dl.UnsupervisedLoader(validation_path, transforms=None) 
    training_dataloader = torch.utils.data.DataLoader(training_loader, batch_size = batch_size, shuffle = True, num_workers = 8, collate_fn = dl.collate_to_list_unsupervised)
    validation_dataloader = torch.utils.data.DataLoader(validation_loader, batch_size = batch_size, shuffle = True, num_workers = 8, collate_fn = dl.collate_to_list_unsupervised)

    regularization_function = cf.curvature_regularization

    training_size = len(training_dataloader.dataset)
    validation_size = len(validation_dataloader.dataset)
    print("Training size: ", training_size)
    print("Validation size: ", validation_size)

    training_cost_before = list()
    training_cost_after = list()
    training_regularization = list()
    validation_cost_before = list()
    validation_cost_after = list()
    validation_regularization = list()

    for i in range(num_epochs):
        training_cost_before.append([])
        training_cost_after.append([])
        training_regularization.append([])
        validation_cost_before.append([])
        validation_cost_after.append([])
        validation_regularization.append([])
        for j in range(num_levels):
            training_cost_before[i].append([])
            training_cost_after[i].append([])
            training_regularization[i].append([])
            validation_cost_before[i].append([])
            validation_cost_after[i].append([])
            validation_regularization[i].append([])
            training_cost_before[i][j] = 0.0
            training_cost_after[i][j] = 0.0
            training_regularization[i][j] = 0.0
            validation_cost_before[i][j] = 0.0
            validation_cost_after[i][j] = 0.0
            validation_regularization[i][j] = 0.0            
            
# Ciclo sugli epoch di addestramento
    for current_epoch in range(num_epochs):
        b_ce = time.time()
        print("Current epoch: ", str(current_epoch + 1) + "/" + str(num_epochs))
        # Training
        #Nella fase di addestramento, vengono definiti i parametri di addestramento e vengono eseguite le iterazioni sugli epoch di addestramento. 
        #Durante ogni epoch, il modello viene addestrato utilizzando il dataset di addestramento e vengono calcolati i costi prima e dopo la registrazione. 
        #Inizia il ciclo di addestramento, scorrendo i dati di addestramento caricati dal training_dataloader
        current_image = 0
        #Itera attraverso il dataloader di validazione, ottenendo coppie di immagini di origine e target.
        for sources, targets in training_dataloader:
            #Questo blocco di codice controlla se deve stampare l'avanzamento.
            #Se il valore di current_image è multiplo di print_step, viene stampato il numero corrente di immagini di addestramento su training_size. Inoltre, incrementa current_image del numero di immagini processate in questo batch.
            if not current_image % print_step:
                print("Training images: ", current_image + 1, "/", training_size)
            current_image += len(sources)
            for k in range(len(sources)):
                source = sources[k]
                target = targets[k]
                current_source = source.to(device).view(1, 1, source.size(0), source.size(1))
                current_target = target.to(device).view(1, 1, target.size(0), target.size(1))
                sources_pyramid = utils.build_pyramid(current_source, num_levels, device=device)
                targets_pyramid = utils.build_pyramid(current_target, num_levels, device=device)
                #Questo ciclo scorre attraverso ogni coppia di immagini di sorgente e target nel batch corrente. 
                # Ogni immagine viene preparata per il calcolo.
                for i in range(num_levels):
                    if i == 0:
                        current_level_displacement_field = torch.zeros(1, 2, targets_pyramid[i].size(2), targets_pyramid[i].size(3)).to(device)
                        current_level_source = sources_pyramid[i]
                    else:
                        current_level_displacement_field = utils.upsample_displacement_fields(current_level_displacement_field, targets_pyramid[i].size(), device=device)
                        current_level_source = utils.warp_tensors(sources_pyramid[i], current_level_displacement_field, device=device)
                    #In questo blocco, vengono preparati i livelli della piramide. 
                    #Viene inizializzata la matrice del campo di spostamento per il livello corrente e viene calcolata la versione "livellata" dell'immagine di sorgente.
                    models[i].train()
                    for inner_iter in range(inner_iterations_per_level[i]):
                        with torch.set_grad_enabled(False):
                            if inner_iter == 0:
                                inner_displacement_field = torch.zeros(1, 2, targets_pyramid[i].size(2), targets_pyramid[i].size(3)).to(device)
                                source_patches, padded_output_size, padding_tuple = utils.unfold(current_level_source, patch_size, stride, device=device)
                                target_patches, _, _ = utils.unfold(targets_pyramid[i], patch_size, stride, device=device)
                            else:
                                warped_source = utils.warp_tensors(current_level_source, inner_displacement_field, device=device)
                                source_patches, padded_output_size, padding_tuple = utils.unfold(warped_source, patch_size, stride, device=device)
                            len_patches = source_patches.size(0)
                            iters = math.ceil(len_patches / number_of_patches)
                            all_displacement_fields = torch.Tensor([]).to(device)
                            real_iters = 0  
                        for j in range(iters):
                            with torch.set_grad_enabled(False):
                                if j == iters - 1:
                                    sp = source_patches[j*number_of_patches:, :, :, :]
                                    tp = target_patches[j*number_of_patches:, :, :, :]
                                else:
                                    sp = source_patches[j*number_of_patches:(j+1)*number_of_patches, :, :, :]
                                    tp = target_patches[j*number_of_patches:(j+1)*number_of_patches, :, :, :]
                                sp = sp + torch.rand(sp.size()).to(device)*0.0000001
                                tp = tp + torch.rand(tp.size()).to(device)*0.0000001
                                if 0 in torch.std(sp, dim=(1, 2, 3)) or 0 in torch.std(tp, dim=(1, 2, 3)):
                                    all_displacement_fields = torch.cat((all_displacement_fields, torch.zeros((sp.size(0), 2, sp.size(2), sp.size(3))).to(device)))
                                    continue
                                real_iters += 1
                            optimizers[i].zero_grad()
                            with torch.set_grad_enabled(True):
                                displacement_fields = models[i](sp, tp)
                                all_displacement_fields = torch.cat((all_displacement_fields, displacement_fields.clone()))
                                tsp = utils.warp_tensors(sp, displacement_fields, device=device)
                                tsp_cf = tsp[:, :, int(stride/2):-int(stride/2), int(stride/2):-int(stride/2)]
                                tp_cf = tp[:, :, int(stride/2):-int(stride/2), int(stride/2):-int(stride/2)]
                                cost = cost_function(tsp_cf, tp_cf, device, **cost_function_params)
                                df_cf = displacement_fields[:, :, int(stride/2):-int(stride/2), int(stride/2):-int(stride/2)]
                                reg = alphas[i]*regularization_function(displacement_fields, device=device)
                                loss = cost + reg
                                loss.backward()
                                optimizers[i].step()
                        with torch.set_grad_enabled(False):
                            all_displacement_fields = utils.fold(all_displacement_fields, padded_output_size, padding_tuple, patch_size, stride, device=device)
                            inner_displacement_field = utils.compose_displacement_fields(inner_displacement_field, all_displacement_fields, device=device)
                    with torch.set_grad_enabled(False):
                        current_level_displacement_field = utils.compose_displacement_fields(current_level_displacement_field, inner_displacement_field, device=device)
                    c_before = cost_function(sources_pyramid[i], targets_pyramid[i], device=device, **cost_function_params)
                    c_after = cost_function(utils.warp_tensors(sources_pyramid[i], current_level_displacement_field, device=device), targets_pyramid[i], device=device, **cost_function_params)
                    c_reg = regularization_function(current_level_displacement_field, device=device)
                    training_cost_before[current_epoch][i] += c_before.item()
                    training_cost_after[current_epoch][i] += c_after.item()
                    training_regularization[current_epoch][i] += c_reg.item()
        # Validation
        current_image = 0
        for sources, targets in validation_dataloader:
            if not current_image % print_step:
                print("Validation images: ", current_image + 1, "/", validation_size)
            current_image += len(sources)
            for k in range(len(sources)):
                #Ottiene l'immagine di origine e l'immagine target corrente.
                source = sources[k]
                target = targets[k]
                #Converte le immagini in tensori e li adatta per essere compatibili con il modello.
                current_source = source.to(device).view(1, 1, source.size(0), source.size(1))
                current_target = target.to(device).view(1, 1, target.size(0), target.size(1))
                # Costruisce le piramidi di immagini di origine e target per ciascun livello di risoluzione.
                sources_pyramid = utils.build_pyramid(current_source, num_levels, device=device)
                targets_pyramid = utils.build_pyramid(current_target, num_levels, device=device)
                #Itera attraverso i diversi livelli della piramide.
                for i in range(num_levels):
                    #primo livello della piramide
                    if i == 0:
                        #Inizializza il campo di spostamento del livello corrente con zeri.
                        current_level_displacement_field = torch.zeros(1, 2, targets_pyramid[i].size(2), targets_pyramid[i].size(3)).to(device)
                        #Imposta l'immagine di origine del livello corrente.
                        current_level_source = sources_pyramid[i]
                    else: 
                        #Altrimenti, se non è il primo livello:
                        #Effettua un upsampling del campo di spostamento del livello corrente per adattarlo alla dimensione del livello corrente della piramide target.
                        current_level_displacement_field = utils.upsample_displacement_fields(current_level_displacement_field, targets_pyramid[i].size(), device=device)
                        #Applica il campo di spostamento corrente all'immagine di origine per ottenere l'immagine di origine trasformata.
                        current_level_source = utils.warp_tensors(sources_pyramid[i], current_level_displacement_field, device=device)
                    # Imposta il modello corrispondente al livello corrente in modalità di valutazione (non addestramento).
                    models[i].eval()
                    #Itera attraverso le iterazioni interne per il livello corrente.
                    for inner_iter in range(inner_iterations_per_level[i]):
                        with torch.set_grad_enabled(False):
                            if inner_iter == 0:
                                inner_displacement_field = torch.zeros(1, 2, targets_pyramid[i].size(2), targets_pyramid[i].size(3)).to(device)
                                #Le immagini di origine e target vengono suddivise in patch (segmenti di immagine) con utils.unfold
                                source_patches, padded_output_size, padding_tuple = utils.unfold(current_level_source, patch_size, stride, device=device)
                                target_patches, _, _ = utils.unfold(targets_pyramid[i], patch_size, stride, device=device)
                            else:
                                warped_source = utils.warp_tensors(current_level_source, inner_displacement_field, device=device)
                                source_patches, padded_output_size, padding_tuple = utils.unfold(warped_source, patch_size, stride, device=device)
                            #Le patch delle immagini di origine e target vengono elaborate iterativamente. 
                            #Se ci sono un numero elevato di patch, il processo può essere suddiviso in più iterazioni.
                            len_patches = source_patches.size(0)
                            iters = math.ceil(len_patches / number_of_patches)
                            all_displacement_fields = torch.Tensor([]).to(device)
                            real_iters = 0  
                        for j in range(iters):
                            with torch.set_grad_enabled(False):
                                if j == iters - 1:
                                    sp = source_patches[j*number_of_patches:, :, :, :]
                                    tp = target_patches[j*number_of_patches:, :, :, :]
                                else:
                                    sp = source_patches[j*number_of_patches:(j+1)*number_of_patches, :, :, :]
                                    tp = target_patches[j*number_of_patches:(j+1)*number_of_patches, :, :, :]
                                #Prima di passare le patch al modello di registrazione, viene aggiunto un piccolo rumore ai valori delle patch per evitare problemi numerici durante il calcolo.
                                #torch.rand(sp.size())  genera un tensore con valori casuali compresi tra 0 e 1, che vengono moltiplicati per una quantità molto piccola e aggiunti alle patch
                                sp = sp + torch.rand(sp.size()).to(device)*0.0000001
                                tp = tp + torch.rand(tp.size()).to(device)*0.0000001
                                #Dopo l'applicazione del rumore, viene effettuato un controllo sulla deviazione standard delle patch per evitare problemi numerici. 
                                #Se la deviazione standard di una patch è zero, viene aggiunto un campo di spostamento nullo per quella patch e il processo continua con la patch successiva.
                                #torch.std(sp, dim=(1, 2, 3)) calcola la deviazione standard lungo le dimensioni spaziali di ciascuna patch. Se uno qualsiasi di questi valori è zero, viene aggiunto un campo di spostamento nullo per quella patch.
                                if 0 in torch.std(sp, dim=(1, 2, 3)) or 0 in torch.std(tp, dim=(1, 2, 3)):
                                    all_displacement_fields = torch.cat((all_displacement_fields, torch.zeros((sp.size(0), 2, sp.size(2), sp.size(3))).to(device)))
                                    continue
                                real_iters += 1
                                #Il modello di registrazione viene quindi utilizzato per calcolare il campo di spostamento tra le patch dell'immagine di origine e quelle dell'immagine target corrente.
                                #models[i] rappresenta il modello di registrazione per il livello corrente
                                # sp e tp  sono le patch dell'immagine di origine e target, rispettivamente. 
                                #Il modello restituisce il campo di spostamento stimato tra le patch.
                                displacement_fields = models[i](sp, tp)
                                #I campi di spostamento calcolati per le varie patch vengono quindi concatenati e utilizzati per aggiornare il campo di spostamento interno. 
                                #all_displacement_fields tiene traccia dei campi di spostamento per tutte le patch elaborare fino a quel punto.
                                all_displacement_fields = torch.cat((all_displacement_fields, displacement_fields.clone()))
                        with torch.set_grad_enabled(False):
                            #I campi di spostamento calcolati per tutte le patch vengono "piegati" (restituiti alla loro dimensione originale nell'immagine) e utilizzati per aggiornare il campo di spostamento interno. 
                            #Questo viene eseguito per garantire che i campi di spostamento siano coerenti con l'intera immagine anziché solo con le patch. 
                            all_displacement_fields = utils.fold(all_displacement_fields, padded_output_size, padding_tuple, patch_size, stride, device=device)
                            inner_displacement_field = utils.compose_displacement_fields(inner_displacement_field, all_displacement_fields, device=device)
                    with torch.set_grad_enabled(False):
                        current_level_displacement_field = utils.compose_displacement_fields(current_level_displacement_field, inner_displacement_field, device=device)
                    #Viene valutato il costo di registrazione prima e dopo l'applicazione del campo di spostamento corrente, così come il costo di regolarizzazione. 
                    c_before = cost_function(sources_pyramid[i], targets_pyramid[i], device=device, **cost_function_params)
                    c_after = cost_function(utils.warp_tensors(sources_pyramid[i], current_level_displacement_field, device=device), targets_pyramid[i], device=device, **cost_function_params)
                    c_reg = regularization_function(current_level_displacement_field, device=device)
                    validation_cost_before[current_epoch][i] += c_before.item()
                    validation_cost_after[current_epoch][i] += c_after.item()
                    validation_regularization[current_epoch][i] += c_reg.item()
        for i in range(num_levels):
            schedulers[i].step()
        e_ce = time.time()
        print("Epoch time: ", e_ce - b_ce, "seconds.")
        print("Estimated time to end epochs: ", (e_ce - b_ce)*(num_epochs - current_epoch - 1), "seconds.")
        for i in range(num_levels):
            # I costi di addestramento e validazione prima e dopo l'applicazione del campo di spostamento, insieme ai costi di regolarizzazione, vengono normalizzati dividendo per le dimensioni rispettive del set di dati di addestramento e di validazione. 
            # Questo viene fatto per consentire un confronto equo tra modelli addestrati su set di dati di dimensioni diverse.
            training_cost_before[current_epoch][i] = training_cost_before[current_epoch][i] / training_size
            training_cost_after[current_epoch][i] = training_cost_after[current_epoch][i] / training_size
            training_regularization[current_epoch][i] = training_regularization[current_epoch][i] / training_size
            validation_cost_before[current_epoch][i] = validation_cost_before[current_epoch][i] / validation_size
            validation_cost_after[current_epoch][i] = validation_cost_after[current_epoch][i] / validation_size
            validation_regularization[current_epoch][i] = validation_regularization[current_epoch][i] / validation_size
            print("Training. Level: ", i, "Epoch: ", current_epoch, " Cost before: ", training_cost_before[current_epoch][i])
            print("Training. Level: ", i, "Epoch: ", current_epoch, " Cost after: ", training_cost_after[current_epoch][i])
            print("Training. Level: ", i, "Epoch: ", current_epoch, " Cost regularization: ", training_regularization[current_epoch][i])
            print("Validation. Level: ", i, "Epoch: ", current_epoch, " Cost before: ", validation_cost_before[current_epoch][i])
            print("Validation. Level: ", i, "Epoch: ", current_epoch, " Cost after: ", validation_cost_after[current_epoch][i])
            print("Validation. Level: ", i, "Epoch: ", current_epoch, " Cost regularization: ", validation_regularization[current_epoch][i])

    #salvataggio dei pesi dei modelli addestrati per ciascun livello.
    if model_save_paths is not None:
        for i in range(num_levels):
            torch.save(models[i].state_dict(), model_save_paths[i])

    for i in range(num_levels):
        parser = lambda a: [a[j][i] for j in range(len(a))]
        plt.figure()
        plt.plot(parser(training_cost_before), color="red", linestyle='-')
        plt.plot(parser(training_cost_after), color="red", linestyle='--')
        plt.plot(parser(validation_cost_before), color="blue", linestyle='-')
        plt.plot(parser(validation_cost_after), color="blue", linestyle='--')
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.legend(["Training Before", "Training After", "Validation Before", "Validation After"])
        plt.title("Level: " + str(i))
        plt.savefig(os.path.join(figures_path, model_name + "_" + str(i) + "_cost_.png"), bbox_inches = 'tight', pad_inches = 0)

        plt.figure()
        plt.plot(parser(training_regularization), color="red", linestyle='-')
        plt.plot(parser(validation_regularization), color="blue", linestyle='-')
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Reg")
        plt.legend(["Training", "Validation"])
        plt.title("Level: " + str(i))
        plt.savefig(os.path.join(figures_path, model_name + "_" + str(i) + "_reg_.png"), bbox_inches = 'tight', pad_inches = 0)
    plt.show()

def visualization(params):
    model_name = params['model_name']
    num_levels = params['num_levels']
    batch_size = 1
    model_save_paths = list()
    models = list()

    #Per ciascun livello del modello, vengono caricate le reti neurali addestrate dai rispettivi percorsi di salvataggio specificati. 
    for i in range(num_levels):
        model_save_paths.append(os.path.join(models_path, model_name + "_level_" + str(i+1)))
        #Vengono creati i data loader per il set di dati di addestramento e di convalida
        models.append(nrn.load_network(device, path=model_save_paths[i]))

    transforms = None
    training_loader = dl.UnsupervisedLoader(training_path, transforms=transforms)
    validation_loader = dl.UnsupervisedLoader(validation_path, transforms=None) 
    training_dataloader = torch.utils.data.DataLoader(training_loader, batch_size = batch_size, shuffle = True, num_workers = 8, collate_fn = dl.collate_to_list_unsupervised)
    validation_dataloader = torch.utils.data.DataLoader(validation_loader, batch_size = batch_size, shuffle = True, num_workers = 8, collate_fn = dl.collate_to_list_unsupervised)

    #Vengono definite due funzioni di costo per valutare la qualità della registrazione: 
    cost_function_1 = cf.ncc_losses_global
    cost_function_params_1 = dict()
    cost_function_2 = cf.mind_loss
    cost_function_params_2 = dict()

    training_size = len(training_dataloader.dataset)
    validation_size = len(validation_dataloader.dataset)
    print("Training size: ", training_size)
    print("Validation size: ", validation_size)

    for sources, targets in training_dataloader:
        for k in range(len(sources)):
            source = sources[k].to(device)
            target = targets[k].to(device)
            #Per ciascuna coppia di immagini di origine e target nel set di dati di addestramento, viene calcolato il campo di spostamento con register(), che utilizza i modelli addestrati. 
            displacement_field = register(source, target, models, params, device=device)
            #l'immagine di origine viene trasformata utilizzando il campo di spostamento calcolato.
            warped_source = utils.warp_tensor(source, displacement_field, device=device)

            #vengono valutati i costi di registrazione utilizzando le funzioni di costo definite.
            print("Initial cost NCC: ", cost_function_1(source.view(1, 1, source.size(0), source.size(1)), target.view(1, 1, source.size(0), source.size(1)), device=device, **cost_function_params_1))
            print("Registered cost NCC: ", cost_function_1(warped_source.view(1, 1, source.size(0), source.size(1)), target.view(1, 1, source.size(0), source.size(1)), device=device, **cost_function_params_1))

            print("Initial cost MIND-SSC: ", cost_function_2(source.view(1, 1, source.size(0), source.size(1)), target.view(1, 1, source.size(0), source.size(1)), device=device, **cost_function_params_2))
            print("Registered cost MIND-SSC: ", cost_function_2(warped_source.view(1, 1, source.size(0), source.size(1)), target.view(1, 1, source.size(0), source.size(1)), device=device, **cost_function_params_2))


            #Le immagini di origine, target e trasformate vengono visualizzate insieme ai campi di spostamento Ux e Uy. 
            
        
            plt.figure(dpi=150)
            plt.subplot(1, 3, 1)
            plt.imshow(source.cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.title("Source")
            plt.subplot(1, 3, 2)
            plt.imshow(target.cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.title("Target")
            plt.subplot(1, 3, 3)
            plt.imshow(warped_source.cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.title("Transformed source")
            
            
            #I campi di spostamento Ux e Uy rappresentano le trasformazioni spaziali che vengono applicate all'immagine di origine per registrare o allineare l'immagine di origine con l'immagine target durante il processo di registrazione non rigida. 
            #Questi campi di spostamento sono matrici bidimensionali delle stesse dimensioni dell'immagine di origine e dell'immagine target.
            #Il campo di spostamento Ux rappresenta lo spostamento orizzontale (lungo l'asse x) che viene applicato a ciascun pixel dell'immagine di origine per allinearla con l'immagine target.
            #Il campo di spostamento Uy rappresenta lo spostamento verticale (lungo l'asse y) che viene applicato a ciascun pixel dell'immagine di origine per allinearla con l'immagine target.
            #Questi campi di spostamento sono solitamente rappresentati come mappe di vettori, dove ciascun vettore rappresenta lo spostamento dal pixel corrispondente nell'immagine di origine al suo corrispondente pixel nell'immagine target. 
            #Ad esempio, se un vettore (u,v) è associato a un pixel nell'immagine di origine, significa che quel pixel viene spostato di u unità lungo l'asse x e di v unità lungo l'asse y per allinearla con l'immagine target.
            
            #Questi campi di spostamento rappresentano le trasformazioni spaziali applicate all'immagine di origine per allinearla con l'immagine target durante il processo di registrazione non rigida.
            #I campi di spostamento sono rappresentati come matrici bidimensionali che hanno le stesse dimensioni dell'immagine di origine e dell'immagine target. 
            #In questo contesto i displacement fields sono rappresentati come tensori bidimensionali con due canali. 
            #Ad esempio, se l'immagine di origine e quella target sono entrambe di dimensioni MxN allora il displacement field sarà un tensore di dimensioni 2xMxN, MxN per l'asse X (Ux), MxN per l'asse Y (Uy)
            
            #Il calcolo dei displacement fields avviene iterando sulle patch delle immagini di origine e target. 
            # Questo approccio permette al modello di registrazione di lavorare in modo efficiente su porzioni locali delle immagini.
            
            plt.figure(dpi=250)
            plt.subplot(1, 2, 1)
            plt.imshow(displacement_field[0].cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.title("Ux")
            plt.subplot(1, 2, 2)
            plt.imshow(displacement_field[1].cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.title("Uy")

            plt.show()
            
            #prova.importo(source, target, warped_source, displacement_field)

def register(source, target, models, params, device='cpu'):
    inner_iterations_per_level = params['inner_iterations_per_level']
    num_levels = params['num_levels']
    patch_size = params['patch_size']
    stride = params['stride']
    number_of_patches = params['number_of_patches']

    with torch.set_grad_enabled(False):
        current_source = source.view(1, 1, source.size(0), source.size(1))
        current_target = target.view(1, 1, target.size(0), target.size(1))
        sources_pyramid = utils.build_pyramid(current_source, num_levels, device=device)
        targets_pyramid = utils.build_pyramid(current_target, num_levels, device=device)
        for i in range(num_levels):
            if i == 0:
                current_level_displacement_field = torch.zeros(1, 2, targets_pyramid[i].size(2), targets_pyramid[i].size(3)).to(device)
                current_level_source = sources_pyramid[i]
            else:
                current_level_displacement_field = utils.upsample_displacement_fields(current_level_displacement_field, targets_pyramid[i].size(), device=device)
                current_level_source = utils.warp_tensors(sources_pyramid[i], current_level_displacement_field, device=device)
            models[i].eval()
            for inner_iter in range(inner_iterations_per_level[i]):
                if inner_iter == 0:
                    inner_displacement_field = torch.zeros(1, 2, targets_pyramid[i].size(2), targets_pyramid[i].size(3)).to(device)
                    source_patches, padded_output_size, padding_tuple = utils.unfold(current_level_source, patch_size, stride, device=device)
                    target_patches, _, _ = utils.unfold(targets_pyramid[i], patch_size, stride, device=device)
                else:
                    warped_source = utils.warp_tensors(current_level_source, inner_displacement_field, device=device)
                    source_patches, padded_output_size, padding_tuple = utils.unfold(warped_source, patch_size, stride, device=device)
                len_patches = source_patches.size(0)
                iters = math.ceil(len_patches / number_of_patches)
                all_displacement_fields = torch.Tensor([]).to(device)
                for j in range(iters):
                    if j == iters - 1:
                        sp = source_patches[j*number_of_patches:, :, :, :]
                        tp = target_patches[j*number_of_patches:, :, :, :]
                    else:
                        sp = source_patches[j*number_of_patches:(j+1)*number_of_patches, :, :, :]
                        tp = target_patches[j*number_of_patches:(j+1)*number_of_patches, :, :, :]
                    displacement_fields = models[i](sp, tp)
                    all_displacement_fields = torch.cat((all_displacement_fields, displacement_fields.clone()))
                all_displacement_fields = utils.fold(all_displacement_fields, padded_output_size, padding_tuple, patch_size, stride, device=device)
                inner_displacement_field = utils.compose_displacement_fields(inner_displacement_field, all_displacement_fields, device=device)
            current_level_displacement_field = utils.compose_displacement_fields(current_level_displacement_field, inner_displacement_field, device=device)
    return current_level_displacement_field[0, :, :, :]

def nonrigid_registration(source, target, models, params, device='cpu'):
    try:
        output_max_size = params['output_max_size']
    except:
        output_max_size = 2048
    with torch.set_grad_enabled(False):
        if max(source.shape) != output_max_size:
            new_shape = utils.calculate_new_shape_max((source.size(0), source.size(1)), output_max_size)
            resampled_source = utils.resample_tensor(source, new_shape, device=device)
            resampled_target = utils.resample_tensor(target, new_shape, device=device)
            displacement_field = register(resampled_source, resampled_target, models, params, device=device)
            displacement_field = utils.upsample_displacement_field(displacement_field, (2, source.size(0), source.size(1)), device=device)
        else:
            resampled_source = source
            resampled_target = target
            displacement_field = register(source, target, models, params, device=device)
        return displacement_field


if __name__ == "__main__":
    # Exemplary training params
    training_params = dict()
    training_params['epochs'] = 2
    training_params['scheduler_rates'] = [0.95, 0.95, 0.95]
    training_params['num_levels'] = 3
    training_params['inner_iterations_per_level'] = [3, 3, 3]
    training_params['stride'] = 128
    training_params['patch_size'] = (256, 256)
    training_params['number_of_patches'] = 32
    training_params['alphas'] = [30, 30, 30]
    training_params['batch_size'] = 1
    training_params['learning_rate'] = 0.001
    training_params['initial_model_name'] = 'nonrigid_2048'
    training_params['cost_function'] = cf.ncc_losses_global
    training_params['cost_function_params'] = dict()
    training_params['cost_function'] = cf.mind_loss
    training_params['cost_function_params'] = dict()
    training_params['model_name'] = "mind_ssc_test"
    training(training_params)

    # Exemplary visualization params
    registration_params = dict()
    registration_params['stride'] = 128
    registration_params['patch_size'] = (256, 256)
    registration_params['number_of_patches'] = 32
    registration_params['num_levels'] = 3
    registration_params['inner_iterations_per_level'] = [3, 3, 3]
    registration_params['model_name'] = "mind_ssc_test"
    visualization(registration_params)