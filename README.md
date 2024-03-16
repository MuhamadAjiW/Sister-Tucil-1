# Tugas Kecil - Paralel Inverse Matrix

## Description
Tugas Kecil IF3230 Sistem Paralel dan Terdistribusi, kelompok SukaSamaKamu

| NIM       | Nama                                     |
|-----------|------------------------------------------|
| 13521045  | Fakhri Muhammad Mahendra                 |
| 13521087  | Razzan Daksana Yoni                      |
| 13521095  | Muhamad Aji Wibisono                     |

## How to Run
### Compile
Untuk mengcompile semua bisa gunakan 
```console
user@user:~/kit-tucil-sister-2024$ make build
```
Atau gunakan `make build_<serial/mpi/openmp/cuda>` untuk mengcompile salah satu saja

### Execute
Untuk mengeksekusi semua bisa gunakan 
```console
user@user:~/kit-tucil-sister-2024$ make exec_all TEST_CASE=<test_case>
```
Atau gunakan `make exec_<serial/mpi/openmp/cuda> TEST_CASE=<test_case>` untuk mengeksekusi salah satu saja

### Combination
Untuk mengcompile sekaligus eksekusi bisa gunakan 
```console
user@user:~/kit-tucil-sister-2024$ make all TEST_CASE=<test_case>
```
Atau gunakan `make <serial/mpi/openmp/cuda> TEST_CASE=<test_case>` untuk mengcompile dan mengeksekusi salah satu saja


## Troubleshooting
Jika kode tidak berjalan dengan baik, konfigurasi eksekusi dan kompilasi dapat diganti pada makefile, khususnya pada bagian variable