# Klasifikacija listova biljaka i bolesti korišćenjem polu nadgledanog učenja i generativnih suparničkih mreža (GAN)

## GAN mreža, odnosno arhitektura mreže se tradicionalno sastoji od 2 dela:
## 1. Generator - mreža koja od ulaznog vektora nasumičnog šuma, generiše podatak u željenom formatu, recimo, slika. Ovo su naravno lažni podaci, jer su generisani iz nasumičnog šuma.

## 2. Diskriminator - mreža koja ima za cilj da klasifikuje slike iz generatora kao lažne, a slike koje dolaze iz skupa za treniranje, kao prave.

## Proces treniranja se sastoji iz 2 faze.
## 1. Treniranje diskriminatora u kojem on dobija kao ulaz prave i lažne slike i na osnovu funkcije greške (između pravih i lažnih), diskriminator uči da razlikuje prave od lažnih slika
## 2. Treniranje generatora. Faza u kojoj uzimamo izlaz iz generatora, dakle lažnu sliku, predajemo diskriminatoru i ocenjujemo koliko ju je on dobro klasifikovao kao pravu ili lažnu. Na osnovu D(G(z)) ažuriramo parametre mreže G (Generator)

## Važna napomena koju uvek treba imati na umu: cilj generatora je da prevari diskriminatora, odnosno da mu podmetne lažnu sliku, a da diskriminator na kraju sa verovatnoćom 1/2 proglasi da je slika lažna ili prava.

## GAN mreža nam tipično služi za generisanje sadržaja, koji prethodno nije viđen. Recimo, generišemo slike pasa, a na osnovu mnoštva postojećih, stvarnih slika pasa.
## U tom kontekstu, nakon što smo trenirali i Generator i Diskriminator, Diskriminator odbacujemo, jer nam je on služio samo da poboljšamo rad generatora. Generator je ono što koristimo nadalje da generišemo neke podatke koji liče na stvarne (slika, zvuk, video)

## Postoje i drugi načini i varijacije na koje se GAN mreže koriste i jedna od njih je predstavljena u ovom radu.
## CGAN - Conditional GAN je GAN u kojem generator generiše sliku sa određenom labelom koja predstavlja klasu - šta je to na slici
### Umesto da Generator izbacuje sliku na osnovu šuma, generatoru se daje i labela, odnosno klasa šta treba da izbaci: sliku psa, mačke, konja ili nekog drugog objekta koji je raspoloživ i labeliram u skupu podataka za treniranje.
## Semi supervised GAN (SGAN), kod ove arhitekture imamo Diskriminator koji nam klasifikuje (D/C) šta je to na slici, na osnovu ulaznog šuma. Kod ove arhitekture, generator odbacujemo, jer nam je on pomogao da poboljšamo diskriminator, da na osnovu ulazne slike "kaže" koja je labela pridružena toj slici.
## Validno je postaviti pitanje smisla D/C GAN mreže, zar to nije običan klasifikator koji će na osnovu ulaznih podataka da nauči da razlikuje klase? Zašto uvodimo GAN kad možemo preko konvolutivne mreže i softmax funkcije da dobijemo klasu slike? Zašto obučavamo 2 mreže (generator i discriminator) kad možemo jednu, CNN?

## Poenta je u tome da imamo mali skup podataka i želimo da generativni deo (generator) pomogne u učenju razlikovanja klase slike

## SGAN se bolje ponaša od CNN kad imamo mali skup podataka, inače u slučaju da imamo dosta podataka, CNN ima prednost.

https://docs.google.com/presentation/d/1WS9vGKvG5_9nvQMEQ9u6mjyjWyNtxYRkMvhvyWMque0/edit?usp=sharing
