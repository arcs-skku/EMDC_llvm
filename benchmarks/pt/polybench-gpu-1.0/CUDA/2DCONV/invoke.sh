#!/bin/bash

for i in $(seq 1 $1);
do
	echo $i > fifo.$i ;
done
