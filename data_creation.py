reviews_u_b_dense_train = defaultdict(dict)
reviews_u_b_dense_val = defaultdict(dict)
reviews_u_b_dense_test = defaultdict(dict)
reviews_u_b_sparse_train = defaultdict(dict)
reviews_u_b_sparse_val = defaultdict(dict)
reviews_u_b_sparse_test = defaultdict(dict)

reviews_b_u_dense_train = defaultdict(dict)
reviews_b_u_dense_val = defaultdict(dict)
reviews_b_u_dense_test = defaultdict(dict)
reviews_b_u_sparse_train = defaultdict(dict)
reviews_b_u_sparse_val = defaultdict(dict)
reviews_b_u_sparse_test = defaultdict(dict)


train_percent = 0.8
val_percent = 0.1
test_percent = 0.1

i = 0
for b in reviews_b_u_dense:    
    l = len(reviews_b_u_dense[b])
    train_l = l * train_percent
    val_l = l * val_percent
    test_l = l * test_percent
    
    for u in reviews_b_u_dense[b]:        
        i +=1
        if i %10 == 9:
            reviews_b_u_dense_test[b][u] = reviews_b_u_dense[b][u]
            reviews_u_b_dense_test[u][b] = reviews_u_b_dense[u][b]
        elif i %10 == 8:    
            reviews_b_u_dense_val[b][u] = reviews_b_u_dense[b][u]
            reviews_u_b_dense_val[u][b] = reviews_u_b_dense[u][b]
        else:
            reviews_b_u_dense_train[b][u] = reviews_b_u_dense[b][u]
            reviews_u_b_dense_train[u][b] = reviews_u_b_dense[u][b]

            
for b in reviews_b_u_sparse:
    
    l = len(reviews_b_u_sparse[b])
    train_l = l * train_percent
    val_l = l * val_percent
    test_l = l * test_percent
    
    for u in reviews_b_u_sparse[b]:
        i +=1
        if i %10 == 9:
            reviews_b_u_sparse_test[b][u] = reviews_b_u_sparse[b][u]
            reviews_u_b_sparse_test[u][b] = reviews_u_b_sparse[u][b]
        elif i % 10 == 8:    
            reviews_b_u_sparse_val[b][u] = reviews_b_u_sparse[b][u]
            reviews_u_b_sparse_val[u][b] = reviews_u_b_sparse[u][b]
        else:
            reviews_b_u_sparse_train[b][u] = reviews_b_u_sparse[b][u]
            reviews_u_b_sparse_train[u][b] = reviews_u_b_sparse[u][b]
            
            
    
    
    
