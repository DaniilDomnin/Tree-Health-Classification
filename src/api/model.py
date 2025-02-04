import torch
import torch.nn as nn


class SimpleTreeHealthModel(nn.Module):
    """Простая модель нейронной сети для классификации состояния деревьев."""

    def __init__(self, embedding_dims, num_classes, num_dim):
        """
        Инициализация слоев модели.

        Args:
            embedding_dims (dict): Словарь, содержащий
             размеры embedding слоев для категориальных признаков.
                                     Ключи - имена категориальных признаков,
                                     значения - кортежи:
                                     (количество уникальных значений,
                                     размерность эмбеддинга).
            num_classes (int): Количество классов для классификации.
            num_dim (int): Размерность входных числовых данных.
        """
        super().__init__()
        self.embedding_layers = nn.ModuleDict(
            {
                name: nn.Embedding(num_embed, embed_dim)
                for name, (num_embed, embed_dim) in embedding_dims.items()
            }
        )
        embedding_output_dim = sum(dims[1] for dims in embedding_dims.values())

        self.cat_seq = nn.Sequential(
            nn.Linear(embedding_output_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.num_seq = nn.Sequential(
            nn.Linear(num_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.final_seq = nn.Sequential(
            nn.Linear(1024 + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, categorical_data, numerical_data):
        """
        Прямой проход модели.

        Args:
            categorical_data (dict): Словарь с категориальными данными.
                                      Ключи - имена категориальных признаков,
                                      значения - тензоры индексов.
            numerical_data (torch.Tensor): Тензор числовых данных.

        Returns:
            torch.Tensor: Выход модели - логиты классов.
        """
        numerical_output = self.num_seq(numerical_data)
        cat_embeddings = []
        for name, data in categorical_data.items():
            cat_embeddings.append(self.embedding_layers[name](data))
        combined_cat_embeddings = torch.cat(cat_embeddings, dim=1)
        categorical_output = self.cat_seq(combined_cat_embeddings)

        combined_features = torch.cat([numerical_output, categorical_output],
                                      dim=1)
        return self.final_seq(combined_features)
