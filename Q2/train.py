import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("mnist-mlp")

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data / 255.0
y = mnist.target.astype(int)
X, y = X[:6000], y[:6000]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def train_and_evaluate(hidden_layer_sizes=(128,), learning_rate_init=0.001, alpha=0.0001, max_iter=20):
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=learning_rate_init,
        alpha=alpha,
        max_iter=max_iter,
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    train_acc = accuracy_score(y_train, model.predict(X_train))
    return model, acc, f1, train_acc


def train_and_log(hidden_layer_sizes=(128,), learning_rate_init=0.001, alpha=0.0001, max_iter=20, run_name=None):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("hidden_layer_sizes", hidden_layer_sizes)
        mlflow.log_param("learning_rate_init", learning_rate_init)
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("max_iter", max_iter)

        model, acc, f1, train_acc = train_and_evaluate(hidden_layer_sizes, learning_rate_init, alpha, max_iter)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("train_loss", model.loss_)

        mlflow.set_tag("team", "data-science")
        mlflow.sklearn.log_model(model, name="model")

        run_id = mlflow.active_run().info.run_id
        print(f"Logged run {run_id}  |  acc={acc:.4f}  f1={f1:.4f}  train_acc={train_acc:.4f}")
        return run_id


sweep = [
    ((64,), 0.001),
    ((128,), 0.001),
    ((256,), 0.001),
    ((64,), 0.01),
    ((128,), 0.01),
    ((128, 64), 0.001),
]

sweep_run_ids = []
for hidden, lr in sweep:
    rid = train_and_log(hidden_layer_sizes=hidden, learning_rate_init=lr,
                        run_name=f"mlp-h{hidden}-lr{lr}")
    sweep_run_ids.append(rid)

print("Sweep run IDs:", sweep_run_ids)

runs_df = mlflow.search_runs(experiment_names=["mnist-mlp"], order_by=["metrics.accuracy DESC"])
best_run = runs_df.iloc[0]
print(f"Best run: {best_run['run_id']}  (accuracy={best_run['metrics.accuracy']:.4f})")
