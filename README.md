# AF

- [ ] changer les attribu actions et is_terminal en fonction
- [ ] changer l'héritage de Double_episode_train pour pouvoir faire un Double_episode_game


## 421

Actuellement les agents sont mal entrainé, je ne sais pas comment est calculé la fonction q mais il est possible qu'elle utilise les états que l'agent n'est pas sensé avoir vu ... -> Il faut trouver une technique pour entrainer le même agent, dans deux contexte séparé, sans que les info ne se collisione.
