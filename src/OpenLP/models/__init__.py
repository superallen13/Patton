from .Graphformer import GraphFormersForLinkPredict, GraphFormersForContextual

AutoModels = {
    'graphformer': GraphFormersForLinkPredict,
    'contextualgraphformer': GraphFormersForContextual
}
