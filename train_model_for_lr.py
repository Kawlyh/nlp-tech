# 数据处理等后
for lr in [1e-5,3e-5,5e-5,7e-5,9e-5]:

    temp = 0.0
    PATH = f"/kaggle/working/per-{i}-{lr}.pt"


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        pearson = metric.compute(predictions=predictions, references=labels)
        #         print(pearson)
        #         print(type(pearson['pearson']))
        #         print(type(temp))
        global temp
        global PATH
        if pearson['pearsonr'] > temp:
            #             torch.save(model.state_dict(),f"/kaggle/working/ckpt{pearson['pearsonr']}.pt")
            torch.save(model, PATH)
            temp = pearson['pearsonr']
        return pearson

    training_args = TrainingArguments(
        output_dir=f'/kaggle/working/checkpoint{i}',  # output directory
        num_train_epochs=50,  # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=4,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=100,
        save_strategy="no",
        evaluation_strategy="epoch",
        learning_rate = lr,
        #         load_best_model_at_end=True,
        #         metric_for_best_model="pearsonr",
        #         greater_is_better=True
    )



    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=tokenized_train,  # training dataset
        eval_dataset=tokenized_val,  # evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    new_model = torch.load(PATH)
    trainer1 = Trainer(model=new_model,
                       args=training_args,  # training arguments, defined above
                       tokenizer=tokenizer,
                       data_collator=data_collator
                       )

    prediction_outputs = trainer1.predict(tokenized_test)
    test_pred = prediction_outputs.predictions[:, -1]
    print(test_pred)
