---
title: nn.Sequential과 nn.ModuleList의 차이점
layout: post
Created: January 24, 2022 4:33 PM
tags:
    - Pytorch
comments: true
use_math: true
sitemap :
  changefreq : daily
  priority : 1.0
---

`nn.ModuleList` 은 forward method가 없지만, `nn.Sequential`은 forward method를 갖고있다. 그래서 `nn.Sequential`을 이용해 여러개의 module을 wrap하여 input을 넣을 수 있다.

`nn.ModuleList`는 단지 python list인데, optimizer를 통해 parameter를 접근하고, train할 수 있기 때문에 유용하다. `nn.Linear`들을 list에 추가하여 만든 것과 동일하다. 단지 좀더 가독성있는 코드를 짤 수 있다.

---

pytorch docs에 있는 ModuleList 예제는 다음과 같다.

```python
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x
```

`nn.Sequential`과 다르게 순서대로 layer를 접근하지 않고, 위와 같은 방식으로 코드를 작성할 수 있다.

---

### reference

1. [ModuleList](https://pytorch.org/docs/1.9.1/generated/torch.nn.ModuleList.html)
2. [When should I use nn.ModuleList and when should I use nn.Sequential?](https://stackoverflow.com/questions/47544051/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential)[Ask Question](https://stackoverflow.com/questions/ask)
