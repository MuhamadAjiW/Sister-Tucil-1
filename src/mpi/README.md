## How it Works
### Cara Kerja Paralisasi
Pertama bagi row tiap matrix ke beberapa proses. Setiap proses bertanggung jawab untuk sejumlah row dari matrix n x 2n.
Pada tiap baris dilakukan penentuan pivot, dan membagi satu baris dengan elemen matrix[n][n] agar baris tersebut memiliki leading one.
Paralelisasi terjadi ketika proses yang telah selesai memroses pivot mengirimkan hasilnya ke proses lain yang membutuhkan hasil tersebut.
Pengiriman bisa terjadi kepada proses yang memiliki baris dibawah pivot ataupun proses yang memiliki baris diatas pivot.

### Cara Program Membagikan Data Antar Proses
Pada program ini, data dibagi dengan cara berikut:
1. Proses 0 membaca input matrix, kemudian membagi matrix menjadi beberapa bagian yang akan dikerjakan oleh proses lain. Hal ini dilakukan dengan perintah MPI_Scatter. Sebelumnya proses 0 mengirimkan ukuran matrix kepada proses lain dengan perintah MPI_Bcast.
2. Untuk tiap proses akan melakukan blocking dengan MPI_Recv untuk menunggu data baris pivot yang dikirimkan oleh proses diatasnya.
3. Proses yang telah selesai menghitung baris pivot akan mengirimkan hasilnya kepada proses lain yang membutuhkan dengan perintah MPI_Send.
4. Proses yang telah selesai mengirim baris pivot akan menunggu data baris dibawah pivot yang dikirimkan oleh proses dibawahnya dengan MPI_Recv.
5. Terakhir dipanggil MPI_Gather untuk mengumpulkan hasil dari tiap proses.

### Alasan Pemilihan Skema Pembagian Data
Skema ini juga memungkinkan proses yang telah selesai menghitung baris pivot untuk mengirimkan hasilnya kepada proses lain yang membutuhkan. Lalu proses-proses yang membuthkan baris pivot tersebut akan memanfaat baris secara paralel dengan proses-proses lainnya. Sehingga program ini dapat berjalan secara paralel.
